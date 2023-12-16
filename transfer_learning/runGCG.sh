#!/bin/bash -l
#$ -pe omp 4
#$ -P ivc-ml
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_c=8.6
#$ -l gpu_memory=48G
#$ -N share_param
#$ -j y
#$ -m ea
module load miniconda
module load pytorch/1.13.1
conda activate /projectnb/ivc-ml/appledora/condaenvs/continual
python /projectnb/cs505ws/students/nimzia/project/CommitMsgEmpirical/train_CBert.py