We have four version of render.

- cudaRenderer_base.cu: slowest for our multi-dimensional scan version.
- cudaRenderer_partition.cu: we do partition for circles.
- cudaRenderer_fast_no_skip.cu: shared memory version.
- cudaRenderer_fastest.cu: fastest for shared memory version with heuristics.