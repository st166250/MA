#!/bin/bash

####

# Define slurm job parameters

####

#SBATCH --job-name=TutorialJob
#SBATCH --cpus-per-task=4
#SBATCH --partition=student
#SBATCH --gpus=1
#SBATCH --time 7-0
#SBATCH --nodelist=node-gpu-01
#SBATCH --error=jobs/job.%J.%x.err
#SBATCH --output=jobs/job.%J.%x.out

echo "Job started on $(date)"
module load cuda/11.8

/home/students/studhoene1/.conda/envs/mypythonenvi/bin/python QualityControlExperiments/train_abdominal2.py