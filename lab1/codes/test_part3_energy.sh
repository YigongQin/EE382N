#!/bin/bash
# rm -r result_matmul
# mkdir result_matmul
for i in 64 # BSIZE1
do
    for j in 128 # BSIZE2
    do
        for k in 1024 # BSZE3
        do
            for (( tt = 0; tt < 20; tt++))
            do
                perf stat -o result_matmul/${i}_${j}_${k}_energy.txt --append -a -e "power/energy-pkg/" -e "power/energy-ram/" ./matmul ${i} ${j} ${k}
            done
        done
    done
done
