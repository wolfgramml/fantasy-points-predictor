#!/bin/bash

#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=rbs-forecasting.out
            
bash --login -cc "conda activate pytorch_forecasting; python rb-timeseries.py"