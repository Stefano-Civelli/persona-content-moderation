#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=Llama70B_Y
#SBATCH --time=80:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --account=a_demartini
#SBATCH -o slurm_output/slurm_Llama70B_Y.o
#SBATCH -e slurm_error/slurm_Llama70B_Y.e
#SBATCH --exclude=bun076

module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate vllm08
module load cuda/12.2.0

python -m src.scripts.8_content_classification_text \
    --model "meta-llama/Llama-3.1-70B-Instruct" \
    --extreme_personas_type "extreme_pos_corners_100_centered" \
    --run_description "Llama70B yoder 100 per corner first run" \
    --dataset_name "yoder" \