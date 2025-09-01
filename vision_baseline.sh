#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=Q_VL32B_baseline
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --account=a_demartini
#SBATCH -o slurm_output/slurm_Q_VL32B_baseline.o
#SBATCH -e slurm_error/slurm_Q_VL32B_baseline.e


module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate vllm08
module load cuda/12.2.0

python -m src.scripts.7_content_classification_img_new \
    --model "Qwen/Qwen2.5-VL-32B-Instruct" \
    --prompt_version "nopersona" \
    --run_description "" \
    --max_samples None \
    --dataset_name "facebook" \