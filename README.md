# Ideology-Based LLMs for Content Moderation

This repository contains the code used in the paper:
**"Ideology-Based LLMs for Content Moderation"**
*Stefano Civelli, Pietro Bernardelle, Nardiena A. Pratama, and Gianluca Demartini.*

---

## 📜 Overview

Large Language Models (LLMs) are increasingly used for content moderation, where neutrality and fairness are critical. This study investigates how assigning a "persona" to an LLM changes its harmful content classification behavior.

While headline metrics like accuracy show little difference, a deeper analysis reveals systematic behavioral shifts. We map synthetic personas to a political compass, select ideologically "extreme" personas, and analyze how their conditioning impacts moderation outcomes.

---

## 📌 Key Findings

1. **Persona Conditioning Shapes Moderation**
   Left-leaning personas flag harmful content more aggressively, while right-leaning personas apply stricter thresholds.

2. **Ideological Cohesion Scales with Model Size**
   Models align more strongly with personas from the same ideological quadrant as model size increases.

3. **Emergent Partisan Bias**
   Larger models are harsher on attacks targeting their own ideological group and more lenient toward attacks on the opposition.

4. **Consistent Across Modalities**
   Most of these effects hold for both text-only and vision-language models, and become more pronounced with scale.

---

## 🧪 Experimental Pipeline

Our methodology, summarized in Figure 1 of the paper, follows four main steps:

1. **Persona Mapping:**
   We prompt six LLMs to take the Political Compass Test (PCT) for 200,000 synthetic personas from PersonaHub, mapping each persona to economic and social coordinates.

2. **Extreme Persona Selection:**
   We select 400 ideologically extreme personas per model (far-left, far-right, and corner cases) to maximize contrast.

3. **Harmful Content Classification:**
   The same LLMs—conditioned on these personas—classify examples from three harmful-content datasets, compared against a no-persona baseline.

4. **Behavioral Analysis:**
   We measure changes in accuracy, precision/recall, intra- and inter-ideological agreement, and partisan bias in politically sensitive cases.

---

## 🤖 Models and Datasets

### Models

We evaluate six open-source, instruction-tuned models to study scaling and architecture effects:

* **Text-Only:**
  `meta-llama/Llama-3.1-8B-Instruct`
  `meta-llama/Llama-3.1-70B-Instruct`
  `Qwen/Qwen2.5-32B-Instruct`

* **Vision-Language:**
  `HuggingFaceM4/Idefics3-8B-Llama3`
  `Qwen/Qwen2.5-VL-7B-Instruct`
  `Qwen/Qwen2.5-VL-32B-Instruct`

Inference uses the `vllm` library for high-throughput execution.

### Datasets

* **Persona Descriptions** – Cleaned `PersonaHub` personas from *Bernardelle et al.* ([Zenodo link](https://zenodo.org/records/16869784))
* **Hate-Identity** – Generic hate speech detection ([Paper](https://aclanthology.org/2022.conll-1.3/))
* **Contextual Abuse Dataset (CAD)** – Political/targeted abuse ([Paper](https://aclanthology.org/2021.naacl-main.182/))
* **Facebook Hateful Memes (FHM)** – Multimodal hate detection ([Challenge page](https://ai.meta.com/tools/hatefulmemes/))


---

## ⚙️ Reproducing the Results

1. **Clone this repository:**

   ```bash
   git clone <repo-url>
   cd persona-content-moderation
   ```

2. **Create and activate a conda environment:**

   ```bash
   conda create -n ib-llms python=3.11
   conda activate persona-moderation
   pip install -r requirements.txt
   ```

3. **Configure the environment:**

   * Edit `models_config.yaml` to set dataset paths and model cache directories.
   * Verify that all datasets are downloaded and accessible from the specified paths.

4. **Run the experiments:**

   * Scripts are numbered in execution order (`1_*.py`, `2_*.py`, ...).
   * Run them sequentially to reproduce the full pipeline.

