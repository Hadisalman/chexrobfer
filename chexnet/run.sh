#!/bin/bash

#SBATCH -o train.log-%j
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive

source /etc/profile

module load anaconda/2020a

python Main.py
