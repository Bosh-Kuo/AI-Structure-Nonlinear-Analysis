#!/bin/bash
#SBATCH --job-name="0.2g 10"
###SBATCH --partition=gtx1070x2
###SBATCH --partition=gtx980
###SBATCH --partition=v100-32g
###SBATCH --gres=gpu:1
#SBATCH --partition=v100-32g
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --time=2-0:00
#SBATCH --chdir=.
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt
###SBATCH --test-only

sbatch_pre.sh

module load opt gcc cuda/9.0 python/gnu-gpu
python3 Disp_02g.py

sbatch_post.sh

