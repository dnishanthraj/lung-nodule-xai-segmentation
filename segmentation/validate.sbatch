#!/bin/bash
#
#SBATCH --job-name=val1 # Job name for tracking
#SBATCH --partition=falcon  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA
# --folder must match a run directory name already present under
# segmentation/model_outputs/ (created by a prior train.py run), e.g.
# NestedUNET_with_augmentation_<timestamp>_<run-id>
python validate.py --name NestedUNET --augmentation True --folder <RUN_FOLDER_NAME>
