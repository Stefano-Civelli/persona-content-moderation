#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=L70B_baseline
#SBATCH --time=5:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --account=a_demartini
#SBATCH -o slurm_output/slurm_L70B_baseline.o
#SBATCH -e slurm_error/slurm_L70B_baseline.e

module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate vllm08
module load cuda/12.2.0

python -m src.scripts.8_content_classification_text \
    --model "meta-llama/Llama-3.1-70B-Instruct" \
    --prompt_version "nopersona" \
    --run_description "" \
    --dataset_name "yoder" \
    --max_samples 10000