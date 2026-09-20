#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-2:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-user=rathodchaitanya11@gmail.com
#SBATCH --mail-type=ALL

cd /home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA
module purge
module load python/3.11 scipy-stack 
module load opencv
module load cuda cudnn
module load nccl
source /home/cjrathod/projects/def-mhassanz/cjrathod/Hamba_env/bin/activate

pip install mamba-ssm==2.2.6.post3