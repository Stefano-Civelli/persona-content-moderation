#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --job-name=QwenVL32
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --account=a_demartini
#SBATCH -o slurm_output/slurm_QwenVL32.o
#SBATCH -e slurm_error/slurm_QwenVL32.e
#SBATCH --exclude=bun116

module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate vllm08
module load cuda/12.2.0

python -m src.scripts.7_content_classification_img_new --model "Qwen/Qwen2.5-VL-32B-Instruct"