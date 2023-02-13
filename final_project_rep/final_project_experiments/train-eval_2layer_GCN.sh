#!/bin/bash

#SBATCH --partition=dgx
#SBATCH --account=undergrad_research
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=./Experiment_outputs/slurm-%j.out

###

# Here's the actual job code. Note: You need to make sure that you execute this from the directory that your python file is located in OR provide an absolute path.

###

container="/data/containers/msoe-pytorch-20.07-py3.sif"

# Command to run inside container

command="python train-eval_2layer_GCN.py"

# Execute singularity container on node.
singularity exec --nv -B /data:/data ${container} ${command}
