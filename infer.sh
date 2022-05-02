#!/bin/sh
#SBATCH --partition=research
#SBATCH --time=15:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=23000
# Memory per node specification is in MB. It is optional. 
# The default limit is 3000MB per core.
#SBATCH --job-name="kanbur-script"
#SBATCH --output=test-srun.out
#SBATCH --mail-user=kanbur@wisc.edu
#SBATCH --mail-type=ALL
#Specifies that the job will be requeued after a node failure.
#The default is that the job will not be requeued.

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

#module load nvidia/cuda/11.6.0
#module list

pip3 install torch torchvision torchaudio torchgeometry opencv-python tqdm

ulimit -s unlimited

python /srv/home/kanbur/double-u-net/euler_inference.py
echo "All Done!"
