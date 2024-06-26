#!/bin/bash

#SBATCH --job-name=depthAnthingPEFT
#SBATCH --output=DepthAnythingPEFT/depthAnthingPEFT.out
#SBATCH --error=DepthAnythingPEFT/depthAnthingPEFT.err
#SBATCH --partition=mundus
#SBATCH --gres=gpu:a100-20:1
#SBATCH --time=07:00:00

python depthest/Underwater_Depth_Estimation/DepthAnythingPEFT/run_training.py