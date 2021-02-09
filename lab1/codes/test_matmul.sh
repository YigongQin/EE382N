#!/bin/bash
rm -r result_matmul
mkdir result_matmul
for i in 32 512 4096
do
    for (( j = 0; j < 20; j++))
    do
        perf stat -o result_matmul/${i}_${i}_${i}_part1.txt --append -e cycles:u,instructions:u,cache-references:u,cache-misses:u ./matmul ${i} ${i} ${i}
        perf stat -o result_matmul/${i}_${i}_${i}_part2.txt --append -e L1-dcache-load:u,L1-dcache-load-misses:u,L1-dcache-stores:u,L1-dcache-store-misses:u ./matmul ${i} ${i} ${i}
    done
done
#use this to submit:  sbatch -p gtx -n 1 -N 1 -t 3:00:00 test_matmul.sh