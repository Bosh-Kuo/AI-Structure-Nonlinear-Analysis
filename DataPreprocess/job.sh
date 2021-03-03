#!/bin/bash
#SBATCH --job-name="DataPreprocess"
###SBATCH --partition=gtx1070x2
###SBATCH --partition=gtx980
###SBATCH --partition=v100-32g
###SBATCH --gres=gpu:1
#SBATCH --partition=1-wing
#SBATCH --ntasks=2
#SBATCH --gres=gpu:0
#SBATCH --time=1-0:00
#SBATCH --chdir=.
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt
###SBATCH --test-only

sbatch_pre.sh

module load opt gcc cuda/9.0 python/gnu
python3 NewDataPreprocess.py

sbatch_post.sh

