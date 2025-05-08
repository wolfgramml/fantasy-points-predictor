#!/bin/bash

#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=combined-forecasting.out
            
bash --login -cc "conda activate pytorch_forecasting; python offensive-players-timeseries.py"