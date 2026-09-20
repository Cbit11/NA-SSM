#!/bin/bash
#SBATCH --job-name=Journal_test
#SBATCH --output=logs/Set5_new_test_results_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-0:10:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-user=rathodchaitanya11@gmail.com
#SBATCH --mail-type=ALL
cd /home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA      #CHANGE THIS    
module purge
module load StdEnv/2023 cudacore/.12.6.3 python/3.11 scipy-stack opencv cuda cudnn
module load nccl/2.27.7
source /home/cjrathod/projects/def-mhassanz/cjrathod/Hamba_env/bin/activate

python test.py  --config /home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA/options/journal_test.yaml


echo "Inference Finished!"