#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --partition=develbooster
### Based on: https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gpu4gc
###SBATCH --partition=gpu
#SBATCH --time=00:15:00

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
###export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
###echo "NODELIST="${SLURM_NODELIST}

export WORLD_SIZE=$SLURM_NTASKS
echo "WORLD_SIZE="$WORLD_SIZE

### Does not work with rank and local rank definition below, as for all processes 0 is handed over.
#export RANK=$SLURM_PROCID
#echo "RANK="$RANK

#export LOCAL_RANK=${SLURM_LOCALID:-$OMPI_COMM_WORLD_LOCAL_RANK}
#echo "LOCAL_RANK="$LOCAL_RANK

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_PORT=12370
echo "MASTER_PORT="$MASTER_PORT

export CUDA_VISIBLE_DEVICES=0,1,2,3

eval "$(/p/project/ccstdl/pieler1/miniconda3/bin/conda shell.bash hook)" # init conda
conda activate pytorch1.10
cd /p/project/ccstdl/pieler1/x-clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun python -u train/train_ddp.py \
--id "test_scaling_gpus/gpu4gc_split_cls_methods_test" \
--path-data-train "/p/scratch/ccstdl/gordon2/CC3M/train/{00000..03318}.tar" \
--save-interval-step 10000 \
--bs 32 \
--lr 1e-4 \
--numw 8 \
--seed 42 \
--loss-over-ranks \
--distributed_backend "PyTorch DDP" \
#--checkdataloading \
#--tb-profiler


### Based on: https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
###SBATCH --constraint=p40&gmem24G
###SBATCH --mem=64gb
###SBATCH --chdir=/scratch/shared/beegfs/your_dir/
###SBATCH --output=/scratch/shared/beegfs/your_dir/%x-%j.out
