#!/bin/bash
#SBATCH --job-name=rand-splits
#SBATCH --time=1-12:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu
# -- SBATCH --dependency=afterok:

set -x

export WORK_DIR="/scratch0/slurm_${SLURM_JOBID}"
export SCRIPT_DIR="/cfarhomes/psando/Documents/coresets-decision-boundaries"

# Set environment for attack
mkdir $WORK_DIR
python3 -m venv ${WORK_DIR}/tmp-env
source ${WORK_DIR}/tmp-env/bin/activate
pip3 install --upgrade pip
pip3 install -r ${SCRIPT_DIR}/requirements.txt

for frac in {0.001,0.01,0.1}
do
    python main.py --runs 5 --split_fraction $frac --workers 4 --batch_size 512 --random_split --no_progress_bar --no_checkpoint_save --no_download_data --cifar_dir /vulcanscratch/psando/cifar-10/
done
