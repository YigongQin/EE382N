import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c

fspace Page {
  rank          : double;
  --
  -- TODO: Add more fields as you need.
  --
  num_out_links : uint64;
  upd_rank      : double;
}

--
-- TODO: Define fieldspace 'Link' which has two pointer fields,
--       one that points to the source and another to the destination.
--
fspace Link(r_src : region(Page),
            r_dst : region(Page)) {
  src : ptr(Page, r_src),
  dst : ptr(Page, r_dst),
}

terra skip_header(f : &c.FILE)
  var x : uint64, y : uint64
  c.fscanf(f, "%llu\n%llu\n", &x, &y)
end

terra read_ids(f : &c.FILE, page_ids : &uint32)
  return c.fscanf(f, "%d %d\n", &page_ids[0], &page_ids[1]) == 2
end

task initialize_graph(r_pages   : region(Page),
                      --
                      -- TODO: Give the right region type here.
                      --
                      r_links   : region(Link(r_pages, r_pages)),
                      damp      : double,
                      num_pages : uint64,
                      filename  : int8[512])
where
  reads writes(r_pages, r_links)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for page in r_pages do
    page.rank = 1.0 / num_pages
    -- TODO: Initialize your fields if you need
    page.num_out_links = 0
    page.upd_rank = (1.0 - damp) / num_pages
  end

  var f = c.fopen(filename, "rb")
  skip_header(f)
  var page_ids : uint32[2]
  for link in r_links do
    regentlib.assert(read_ids(f, page_ids), "Less data that it should be")
    var src_page = dynamic_cast(ptr(Page, r_pages), page_ids[0])
    var dst_page = dynamic_cast(ptr(Page, r_pages), page_ids[1])
    --
    -- TODO: Initialize the link with 'src_page' and 'dst_page'
    --
    link.src = src_page
    link.dst = dst_page
    link.src.num_out_links += 1
  end
  c.fclose(f)
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Graph initialization took %.4f sec\n", (ts_stop - ts_start) * 1e-6)
end

--
-- TODO: Implement PageRank. You can use as many tasks as you want.
--
task update_ranks(r_src   : region(Page),
                  r_dst   : region(Page),
                  r_links : region(Link(r_src, r_dst)),
                  damp    : double)
where
  reads(r_links.{src, dst}),
  reads(r_src.{rank, num_out_links}),
  reduces +(r_dst.upd_rank)
do
  for link in r_links do
    --c.printf("%lf\n", link.src.rank)
    --c.printf("%d\n", link.src.num_out_links)
    --c.printf("%lf\n", damp * link.src.rank / link.src.num_out_links)
    --c.printf("%lf\n", link.dst.upd_rank)
    link.dst.upd_rank += damp * (link.src.rank / link.src.num_out_links)
    --c.printf("%lf\n", link.dst.upd_rank)
  end
end

task calc_error(r_pages   : region(Page),
                num_pages : uint64,
                damp      : double)
where
  reads writes(r_pages.{rank, upd_rank})
do
  var sum_delta_sq : double = 0.0
  for page in r_pages do
    var delta = page.upd_rank - page.rank
    sum_delta_sq += delta * delta
    page.rank = page.upd_rank
    page.upd_rank = (1.0 - damp) / num_pages
  end
  return sum_delta_sq
end

task time_step(config  : PageRankConfig,
               r_pages : region(Page),
               r_links : region(Link(wild, wild)),
               p_pages : partition(disjoint, r_pages, ispace(int1d)),
               p_links : partition(disjoint, r_links, ispace(int1d)),
--               p_src   : partition(disjoint, r_pages, ispace(int1d)),
--               p_dst   : partition(aliased, r_pages, ispace(int1d)))
               p_src   : partition(aliased, r_pages, ispace(int1d)),
               p_dst   : partition(disjoint, r_pages, ispace(int1d)))
where
  reads writes(r_pages.{rank, upd_rank, num_out_links}),
  reads(r_links.{src, dst})
do
  __demand(__parallel)
  for i = 0, config.parallelism do
    update_ranks(p_src[i], p_dst[i], p_links[i], config.damp)
  end

  var sum_delta_sq : double = 0.0
  __demand(__parallel)
  for i = 0, config.parallelism do
    sum_delta_sq += calc_error(p_pages[i], config.num_pages, config.damp)
  end

  return sum_delta_sq
end

task dump_ranks(r_pages  : region(Page),
                filename : int8[512])
where
  reads(r_pages.rank)
do
  var f = c.fopen(filename, "w")
  for page in r_pages do c.fprintf(f, "%g\n", page.rank) end
  c.fclose(f)
end

task toplevel()
  var config : PageRankConfig
  config:initialize_from_command()
  c.printf("**********************************\n")
  c.printf("* PageRank                       *\n")
  c.printf("*                                *\n")
  c.printf("* Number of Pages  : %11lu *\n",  config.num_pages)
  c.printf("* Number of Links  : %11lu *\n",  config.num_links)
  c.printf("* Damping Factor   : %11.4f *\n", config.damp)
  c.printf("* Error Bound      : %11g *\n",   config.error_bound)
  c.printf("* Max # Iterations : %11u *\n",   config.max_iterations)
  c.printf("* # Parallel Tasks : %11u *\n",   config.parallelism)
  c.printf("**********************************\n")

  -- Create a region of pages
  var r_pages = region(ispace(ptr, config.num_pages), Page)
  --
  -- TODO: Create a region of links.
  --       It is your choice how you allocate the elements in this region.
  --
  var r_links = region(ispace(ptr, config.num_links), Link(wild, wild))

  --
  -- TODO: Create partitions for links and pages.
  --       You can use as many partitions as you want.
  --
  var colors = ispace(int1d, config.parallelism)

  -- independent (equal) partitions of source pages (nodes)
  var p_pages = partition(equal, r_pages, colors)

  -- Initialize the page graph from a file
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input)

  -- dependent partitions of edges/dst derived from src partitions
--  var p_src = p_pages
--  var p_links = preimage(r_links, p_pages, r_links.src)
--  var p_dst = image(r_pages, p_links, r_links.dst)

  var p_dst = p_pages
  var p_links = preimage(r_links, p_dst, r_links.dst)
  var p_src = image(r_pages, p_links, r_links.src)

  var num_iterations = 0
  var converged = false
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_start = c.legion_get_current_time_in_micros()
  while not converged do
    num_iterations += 1
    --
    -- TODO: Launch the tasks that you implemented above.
    --       (and of course remove the break statement here.)
    --
    var norm = time_step(config, r_pages, r_links, p_pages, p_links, p_src, p_dst)
--    update_ranks(r_pages, r_pages, r_links, config.damp)
--    var norm = calc_error(r_pages, config.num_pages, config.damp)

    converged = (c.sqrt(norm) <= config.error_bound) or (num_iterations >= config.max_iterations)
  end
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)
