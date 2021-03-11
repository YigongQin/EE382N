#!/bin/bash
#SBATCH -J render           # job name
#SBATCH -o render.o%j       # expands to jobID
#SBATCH -N 1                # total number of node
#SBATCH -n 1                # total number of mpi tasks requested
#SBATCH -p gtx              # queue (partition) — normal, development, etc.
#SBATCH -t 00:30:00         # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=someemailaddrofyours@gmail.com
#SBATCH --mail-type=begin   # email me when the job starts
#SBATCH --mail-type=end     # email me when the job finishes
#SBATCH -A EE382N-20
module load gcc
module load cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/8.0/lib64
./render -r cuda -b 0:4 -f rgbtest_cuda -s 768 rgb
./checker.pl 
