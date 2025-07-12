ENV_NAME = vllm08

setup:
	conda create -y -n $(ENV_NAME) python=3.11
	conda run -n $(ENV_NAME) pip install -r requirements.txt

install:
	conda run -n $(ENV_NAME) pip install -r requirements.txt

install_nightly_vllm:
	pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

extreme_personas:
	python -m src.scripts.5_extreme_personas --use_center

plot_ext_personas:
	python -m src.plots.plot_extreme_personas --use_center

run_vision:
	python -m src.scripts.7_content_classification_img_new --dataset_name "multioff"

run_text:
	python -m src.scripts.8_content_classification_text

agreement:
	python -m src.scripts.9_agreement

embedding:
	python -m src.scripts.11_persona_behavioural_embedding

embedding_2:
	python -m src.scripts.11_persona_behavioural_embedding_2

subdata:
	python -m src.scripts.12_subdata_analysis

metadata:
	python -m utils.print_metadata

run_script:
	python -m src.scripts.$(script)

a100:
	salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G --job-name=TinyGPU --time=01:30:00 --partition=gpu_cuda --qos=gpu --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1 --account=a_demartini srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l

clean_cache:
	rm -r ~/.cache