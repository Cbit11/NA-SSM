#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-3:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mail-user=rathodchaitanya11@gmail.com
#SBATCH --mail-type=ALL
cd /home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA
module purge
module load StdEnv/2023 cudacore/.12.6.3 python/3.11 scipy-stack opencv cuda cudnn
module load nccl/2.27.7
source /home/cjrathod/projects/def-mhassanz/cjrathod/Hamba_env/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 10000-60000 -n 1)
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
wandb login $API_KEY 
echo "Step 1: Copying data to local SSD ($SLURM_TMPDIR)..."
TRAIN_HR="/home/cjrathod/scratch/Download_df2k/datasets/DF2K/DF2K_train_HR"
TRAIN_LR="/home/cjrathod/scratch/Download_df2k/datasets/DF2K/DF2K_train_LR_bicubic/X2"
VAL_HR="/home/cjrathod/scratch/Download_df2k/datasets/DIV2K/DIV2K_valid_HR"
VAL_LR="/home/cjrathod/scratch/Download_df2k/datasets/DIV2K/DIV2K_valid_LR_bicubic/X2"


cp -ru $TRAIN_HR $SLURM_TMPDIR/train_hr
cp -ru $TRAIN_LR $SLURM_TMPDIR/train_lr_x2
cp -ru $VAL_HR $SLURM_TMPDIR/val_hr
cp -ru $VAL_LR $SLURM_TMPDIR/val_lr_x2

echo "Data copy complete. Contents of local scratch:"
ls -lh $SLURM_TMPDIR
echo "SLURM Job Info: MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"
echo "Launching Distributed PyTorch training..."
# --- WANDB OPTIMIZATIONS ---
export WANDB_DIR=/home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA/wandb_logs/wandb_logs
export WANDB_CACHE_DIR=/home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA/wandb_logs/cache
export WANDB_SERVICE_WAIT=300
srun python main.py \
    --config ./options/journal_model_ablation2.yaml \
    --train_hr_pth $SLURM_TMPDIR/train_hr \
    --train_lr_pth $SLURM_TMPDIR/train_lr_x2 \
    --val_hr_pth $SLURM_TMPDIR/val_hr \
    --val_lr_pth $SLURM_TMPDIR/val_lr_x2\
    --checkpoint_folder /home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA/chkpts/ablations_no_nat