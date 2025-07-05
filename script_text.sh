#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=Qwen32
#SBATCH --time=50:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --account=a_demartini
#SBATCH -o slurm_output/slurm_Qwen32.o
#SBATCH -e slurm_error/slurm_Qwen32.e
#SBATCH --exclude=bun074

module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate vllm08
module load cuda/12.2.0

python -m src.scripts.8_content_classification_text --model "Qwen/Qwen2.5-32B-Instruct" --extreme_personas_type "extreme_pos_corners_100"