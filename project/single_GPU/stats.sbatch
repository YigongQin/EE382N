#!/bin/bash
#SBATCH -J render           # job name
#SBATCH -o render.o%j       # expands to jobID
#SBATCH -N 1                # total number of node
#SBATCH -n 1                # total number of mpi tasks requested
#SBATCH -p gtx              # queue (partition) — normal, development, etc.
#SBATCH -t 00:05:00         # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=siyuan.ma@utexas.edu
#SBATCH --mail-type=begin   # email me when the job starts
#SBATCH --mail-type=end     # email me when the job finishes
#SBATCH -A EE382N-20-Parallelis
module load gcc
module load cuda
./phase_field input_many 1 >output_1box.o
./phase_field input_many 4 >output_4box.o
./phase_field input_many 8 >output_8box.o
./phase_field input_many 12 >output_12box.o