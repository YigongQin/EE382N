import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c

fspace Page {
  rank         : double;
  --
  -- TODO: Add more fields as you need.
  -- num_out: connection with others as denominator
  -- neighb_src: updated rank
  num_out : uint64,
  neighb_src : double,
}


--
-- TODO: Define fieldspace 'Link' which has two pointer fields,
--       one that points to the source and another to the destination.
--
-- fspace Link(...) { ... }
fspace Link(r_src: region(Page), r_dst: region(Page))
{
  src_page : ptr(Page, r_src),
  dst_page: ptr(Page, r_dst),
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
    page.num_out = 0
    page.neighb_src = 0.0
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
    link.src_page = src_page
    link.dst_page = dst_page 
    link.src_page.num_out += 1

  end
  c.fclose(f)
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Graph initialization took %.4f sec\n", (ts_stop - ts_start) * 1e-6)
end

--
-- TODO: Implement PageRank. You can use as many tasks as you want.
--
-- Comment: Have been merged into initialize_graph
-- task1: find the number of outgoing links for every page p
-- for every edge, find the source, add its num_out
task count_out_link(r_pages : region(Page),r_links : region(Link(r_pages, r_pages)))
where reads (r_pages, r_links), reduces+(r_pages.num_out) do
  for link in r_links do
    link.src_page.num_out += 1
  end  
  --for page in r_pages do
  --  c.printf("number of out link %d \n", page.num_out)
  --end 
 
end


-- task2: for evevry link, calculate the src contribution to the dest, add to dest.neighb_src
-- has dependency on task1

-- error is here
task neighbor_contribution(r_src : region(Page),
                           r_dst : region(Page),
                           r_links : region(Link(r_src, r_dst))
)
where reads (r_src.{rank, num_out}, r_links.{src_page, dst_page}), reduces+(r_dst.neighb_src) do
  for link in r_links do
    link.dst_page.neighb_src += link.src_page.rank/link.src_page.num_out
  end
  --for page in r_pages do
  --  c.printf("neighbor weights %f \n", page.neighb_src)
  --end
end

--task3: update the page rank and check residual
task update_page_rank(r_pages : region(Page), damp : double,
                     num_pages : uint64, error_bound : double) 
where
  reads writes(r_pages.{rank, neighb_src})
do
  var residual : double = 0.0
  var temp : double
  for page in r_pages do
     temp = (1.0-damp)/num_pages + damp*page.neighb_src
     residual += (temp-page.rank)*(temp-page.rank)
     page.rank = temp
     -- remeber to clean up the neighb_src
     page.neighb_src = 0.0
  end
  
  return residual
  --if residual<error_bound*error_bound then
  --return true
  --else
  --return false
  --end
end

--task4: update all in one piece
task update_all(config  : PageRankConfig,
                r_pages: region(Page),
                r_links: region(Link(wild, wild)),
                p_pages: partition(disjoint, r_pages, ispace(int1d)),
                p_links: partition(disjoint, r_links, ispace(int1d)),
                p_src  : partition(disjoint, r_pages, ispace(int1d)),
                p_dst  : partition(aliased, r_pages, ispace(int1d))
)

where reads writes(r_pages.{rank, neighb_src, num_out}),
      reads (r_links.{src_page, dst_page})
do
  __demand(__index_launch)
  for i = 0, config.parallelism do
    neighbor_contribution(p_src[i], p_dst[i], p_links[i])
  end

  var err : double = 0.0
  __demand(__index_launch)
  for i = 0, config.parallelism do
    err += update_page_rank(p_pages[i], config.damp, config.num_pages, config.error_bound)
  end

  return err
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
  var r_links = region(ispace(ptr, config.num_links), Link(r_pages, r_pages))

  --
  -- TODO: Create partitions for links and pages.
  --       You can use as many partitions as you want.
  -- Add partitions idx region

  var num_p : uint32 = config.parallelism
  var colors = ispace(int1d, num_p)

  -- Initialize the page graph from a file
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input)

  -- independent partitions for nodes(source pages)
  var p_pages = partition(equal, r_pages, colors)

  -- dependent partitions for nodes(dst) and links
  var p_src   = p_pages
  var p_links = preimage(r_links, p_src, r_links.src_page)
  var p_dst   = image(r_pages, p_links, r_links.dst_page)

  var num_iterations = 0
  var converged = false
  var err: double = 0.0
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_start = c.legion_get_current_time_in_micros()
  count_out_link(r_pages,r_links)
  while not converged do
    num_iterations += 1
    --
    -- TODO: Launch the tasks that you implemented above.
    --       (and of course remove the break statement here.)
    --
    err = update_all(config, r_pages, r_links, p_pages, p_links, p_src, p_dst)

    if num_iterations== config.max_iterations or err<config.error_bound*config.error_bound then
       break
    end
  end
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)
