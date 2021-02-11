#!/bin/bash
#input="result_matmul/32_32_32_part1.txt"
for input in "result_matmul"/*_part*.txt
do
cycles="0"
instructions="0"
cache_references="0"
cache_misses="0"
L1_dcache_load="0"
L1_dcache_load_misses="0"
L1_dcache_stores="0"
L1_dcache_store_misses="0"
time="0"
while IFS= read -r line
do
    #echo "$line"
    arr=($line)
    item=${arr[1]}
    #echo "$item"

    case "$item" in

    "seconds")
        #echo "${arr[0]}"
        time="${arr[0]} + ${time}"
        ;;

    "cycles:u")
        #echo "${arr[0]}"
        cycles=${arr[0]}+${cycles}
        ;;

    "instructions:u")
        #echo "${arr[0]}"
        instructions=${arr[0]}+${instructions}
        ;;

    "cache-references:u")
        #echo "${arr[0]}"
        cache_references=${arr[0]}+${cache_references}
        ;;

    "cache-misses:u")
        #echo "${arr[0]}"
        cache_misses=${arr[0]}+${cache_misses}
        ;;

    "L1-dcache-load")
        #echo "${arr[0]}"
        L1_dcache_load=${arr[0]}+${L1_dcache_load}
        ;;

    "L1-dcache-load-misses")
        #echo "${arr[0]}"
        L1_dcache_load_misses=${arr[0]}+${L1_dcache_load_misses}
        ;;

    "L1-dcache-stores")
        #echo "${arr[0]}"
        L1_dcache_stores=${arr[0]}+${L1_dcache_stores}
        ;;

    "L1-dcache-store-misses")
        #echo "${arr[0]}"
        L1_dcache_store_misses=${arr[0]}+${L1_dcache_store_misses}
        ;;

    "*")
        ;;
    esac
done < "$input"
echo "total cycles = "$((cycles))>> ${input}
echo "scale=2 ; $((cycles)) / 20" | bc >> ${input}
echo "total instructions = "$((instructions))>> ${input}
echo "scale=2 ; $((instructions)) / 20" | bc >> ${input}
echo "total cache_references = "$((cache_references))>> ${input}
echo "scale=2 ; $((cache_references)) / 20" | bc >> ${input}
echo "total cache_misses = "$((cache_misses))>> ${input}
echo "scale=2 ; $((cache_misses)) / 20" | bc >> ${input}
echo "total L1_dcache_load = "$((L1_dcache_load))>> ${input}
echo "scale=2 ; $((L1_dcache_load)) / 20" | bc >> ${input}
echo "total L1_dcache_load_misses = "$((L1_dcache_load_misses))>> ${input}
echo "scale=2 ; $((L1_dcache_load_misses)) / 20" | bc >> ${input}
echo "total L1_dcache_stores = "$((L1_dcache_stores))>> ${input}
echo "scale=2 ; $((L1_dcache_storess)) / 20" | bc >> ${input}
echo "total L1_dcache_store_misses = "$((L1_dcache_store_misses))>> ${input}
echo "scale=2 ; $((L1_dcache_store_misses)) / 20" | bc >> ${input}
# echo "total time = "$((time)) >> ${input}
time="$(echo "$time" | bc -l)"
echo "total time = ${time}">> ${input}
average="$(echo "${time}/20" | bc -l)"
average="$(echo "$average" | sed -e 's/^\(.*\..*[^0]\)0*/\1/' -e p)"
echo "time = ${average}">> ${input}
# echo "scale=2 ; $((time)) / 20" | bc -l >> ${input}
done
