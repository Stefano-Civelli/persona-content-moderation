2025-07-02 23:53:02,174 - INFO - ======================================================================
2025-07-02 23:53:02,175 - INFO - Extreme personas path: data/results/extreme_pos_personas/Qwen2.5-VL-7B-Instruct/extreme_pos_corners.pkl
2025-07-02 23:53:02,175 - INFO - Output path: data/results/img_classification/Qwen2.5-VL-7B-Instruct/20250702_235302.json
2025-07-02 23:53:02,175 - INFO - Using model: Qwen/Qwen2.5-VL-7B-Instruct
2025-07-02 23:53:02,175 - INFO - ======================================================================

2025-07-02 23:53:02,233 - INFO - ======================================================================
2025-07-02 23:53:02,233 - INFO - VLLM Parameters:
2025-07-02 23:53:02,233 - INFO - {'model': 'Qwen/Qwen2.5-VL-7B-Instruct', 'tokenizer_mode': 'auto', 'trust_remote_code': True, 'enforce_eager': False, 'dtype': 'auto', 'gpu_memory_utilization': 0.95, 'tensor_parallel_size': 1, 'enable_prefix_caching': True, 'disable_log_stats': True, 'download_dir': '/scratch/user/uqscivel/HF-CACHE', 'limit_mm_per_prompt': {'image': 1}, 'max_model_len': 2048, 'max_num_seqs': 500}
2025-07-02 23:53:02,233 - INFO - ======================================================================

Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:01<00:05,  1.45s/it]
Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:06<00:10,  3.61s/it]
Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:11<00:08,  4.32s/it]
Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:16<00:04,  4.66s/it]
Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:22<00:00,  4.87s/it]
Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:22<00:00,  4.43s/it]

2025-07-02 23:54:44,741 - INFO - ======================================================================
2025-07-02 23:54:44,741 - INFO - Loading extreme personas and building prompts...
2025-07-02 23:54:44,744 - INFO - Loaded 60 extreme personas.
2025-07-02 23:54:44,745 - INFO - Sample extreme persona: {'persona_id': 134300, 'description': "a newspaper editor known for his conservative bias and strong influence over the publication's political stance.", 'pos': 'top_right'}
  0%|          | 0/60 [00:00<?, ?it/s]100%|██████████| 60/60 [00:00<00:00, 275941.05it/s]
2025-07-02 23:54:44,747 - INFO - Generated 60 prompts.
2025-07-02 23:54:44,748 - INFO - Sample prompt: {'persona_id': 134300, 'persona_pos': 'top_right', 'persona': "a newspaper editor known for his conservative bias and strong influence over the publication's political stance.", 'prompt': 'You are an expert content analyst specializing in identifying hate. Your task is to analyze the provided meme strictly from the perspective of a newspaper editor known for his conservative bias and strong influence over the publication\'s political stance..\n\nBased on your analysis, you must generate a single, valid JSON object that adheres to the following schema and logic. Do not output any text other than the JSON object.\n\n**Analysis Logic:**\n1.  **Perspective:** Embody the persona of a newspaper editor known for his conservative bias and strong influence over the publication\'s political stance.. Your entire analysis must reflect their likely values, biases, worldview and sensitivity level to hateful content\n2.  **Classification:** First, determine if the persona would consider the meme to be hate speech. This determines the value of `is_hate_speech`.\n3.  **Conditional Analysis:**\n    - **If `is_hate_speech` is "true"**: You MUST identify the single, *most prominent* `target_group` and the single, *primary* `attack_method` used.\n    - **If `is_hate_speech` is "false"**: The `target_group` and `attack_method` fields MUST be `null`.\n\n**JSON Schema:**\n- `is_hate_speech` (string): Must be one of ["true", "false"].\n- `target_group` (string or null): If hateful, must be one of ["disability", "race", "religion", "nationality", "sex"]. Otherwise, it must be `null`.\n- `attack_method` (string or null): If hateful, must be one of ["contempt", "mocking", "inferiority", "slurs", "exclusion", "dehumanizing", "inciting_violence"]. Otherwise, it must be `null`.\n\n**Example 1 (Hateful):**\n{\n  "is_hate_speech": "true",\n  "target_group": "race",\n  "attack_method": "mocking"\n}\n\n**Example 2 (Not Hateful):**\n{\n  "is_hate_speech": "false",\n  "target_group": null,\n  "attack_method": null\n}\n\nNow, analyze the image from the persona\'s perspective and generate the JSON object.\n'}

2025-07-02 23:54:44,751 - INFO - ======================================================================

2025-07-02 23:54:44,751 - INFO - Dataset size: 22080
2025-07-02 23:54:44,752 - INFO - Processing 22080 items in batches of 2000...
  0%|          | 0/12 [00:00<?, ?it/s]2025-07-02 23:55:04,913 - INFO - ======================================================================
2025-07-02 23:55:04,914 - DEBUG - unformatted prompts: ['You are an expert content analyst specializing in identifying hate. Your task is to analyze the provided meme strictly from the perspective of a newspaper editor known for his conservative bias and strong influence over the publication\'s political stance..\n\nBased on your analysis, you must generate a single, valid JSON object that adheres to the following schema and logic. Do not output any text other than the JSON object.\n\n**Analysis Logic:**\n1.  **Perspective:** Embody the persona of a newspaper editor known for his conservative bias and strong influence over the publication\'s political stance.. Your entire analysis must reflect their likely values, biases, worldview and sensitivity level to hateful content\n2.  **Classification:** First, determine if the persona would consider the meme to be hate speech. This determines the value of `is_hate_speech`.\n3.  **Conditional Analysis:**\n    - **If `is_hate_speech` is "true"**: You MUST identify the single, *most prominent* `target_group` and the single, *primary* `attack_method` used.\n    - **If `is_hate_speech` is "false"**: The `target_group` and `attack_method` fields MUST be `null`.\n\n**JSON Schema:**\n- `is_hate_speech` (string): Must be one of ["true", "false"].\n- `target_group` (string or null): If hateful, must be one of ["disability", "race", "religion", "nationality", "sex"]. Otherwise, it must be `null`.\n- `attack_method` (string or null): If hateful, must be one of ["contempt", "mocking", "inferiority", "slurs", "exclusion", "dehumanizing", "inciting_violence"]. Otherwise, it must be `null`.\n\n**Example 1 (Hateful):**\n{\n  "is_hate_speech": "true",\n  "target_group": "race",\n  "attack_method": "mocking"\n}\n\n**Example 2 (Not Hateful):**\n{\n  "is_hate_speech": "false",\n  "target_group": null,\n  "attack_method": null\n}\n\nNow, analyze the image from the persona\'s perspective and generate the JSON object.\n', 'You are an expert content analyst specializing in identifying hate. Your task is to analyze the provided meme strictly from the perspective of a social media influencer who shares right-wing political ideologies and promotes tough-on-crime policies..\n\nBased on your analysis, you must generate a single, valid JSON object that adheres to the following schema and logic. Do not output any text other than the JSON object.\n\n**Analysis Logic:**\n1.  **Perspective:** Embody the persona of a social media influencer who shares right-wing political ideologies and promotes tough-on-crime policies.. Your entire analysis must reflect their likely values, biases, worldview and sensitivity level to hateful content\n2.  **Classification:** First, determine if the persona would consider the meme to be hate speech. This determines the value of `is_hate_speech`.\n3.  **Conditional Analysis:**\n    - **If `is_hate_speech` is "true"**: You MUST identify the single, *most prominent* `target_group` and the single, *primary* `attack_method` used.\n    - **If `is_hate_speech` is "false"**: The `target_group` and `attack_method` fields MUST be `null`.\n\n**JSON Schema:**\n- `is_hate_speech` (string): Must be one of ["true", "false"].\n- `target_group` (string or null): If hateful, must be one of ["disability", "race", "religion", "nationality", "sex"]. Otherwise, it must be `null`.\n- `attack_method` (string or null): If hateful, must be one of ["contempt", "mocking", "inferiority", "slurs", "exclusion", "dehumanizing", "inciting_violence"]. Otherwise, it must be `null`.\n\n**Example 1 (Hateful):**\n{\n  "is_hate_speech": "true",\n  "target_group": "race",\n  "attack_method": "mocking"\n}\n\n**Example 2 (Not Hateful):**\n{\n  "is_hate_speech": "false",\n  "target_group": null,\n  "attack_method": null\n}\n\nNow, analyze the image from the persona\'s perspective and generate the JSON object.\n']
2025-07-02 23:55:04,914 - DEBUG - images: [<PIL.Image.Image image mode=RGB size=825x549 at 0x7FED960AA050>, <PIL.Image.Image image mode=RGB size=825x549 at 0x7FED960B0ED0>]
2025-07-02 23:55:04,914 - DEBUG - batch_labels: [{'hate': ['not_hateful'], 'pc': ['pc_empty'], 'attack': ['attack_empty']}, {'hate': ['not_hateful'], 'pc': ['pc_empty'], 'attack': ['attack_empty']}]
2025-07-02 23:55:04,914 - DEBUG - item_ids: ['img/05349.png', 'img/05349.png']
2025-07-02 23:55:04,914 - DEBUG - persona_ids: [134300, 53815]
2025-07-02 23:55:04,914 - DEBUG - persona_pos: ['top_right', 'top_right']
2025-07-02 23:55:04,914 - INFO - ======================================================================

Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.

Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:22<12:38:57, 22.78s/it, est. speed input: 46.88 toks/s, output: 1.40 toks/s][A
Processed prompts:   1%|          | 17/2000 [00:24<33:58,  1.03s/it, est. speed input: 753.26 toks/s, output: 22.66 toks/s][A
Processed prompts:   2%|▏         | 35/2000 [00:25<14:40,  2.23it/s, est. speed input: 1472.07 toks/s, output: 44.28 toks/s][A
Processed prompts:   3%|▎         | 53/2000 [00:26<08:54,  3.64it/s, est. speed input: 2118.13 toks/s, output: 63.69 toks/s][A
Processed prompts:   4%|▎         | 70/2000 [00:27<06:22,  5.05it/s, est. speed input: 2512.80 toks/s, output: 80.24 toks/s][A
Processed prompts:   4%|▍         | 90/2000 [00:29<04:38,  6.87it/s, est. speed input: 2842.27 toks/s, output: 98.52 toks/s][A
Processed prompts:   6%|▌         | 117/2000 [00:30<03:16,  9.57it/s, est. speed input: 3398.66 toks/s, output: 121.88 toks/s][A
Processed prompts:   8%|▊         | 150/2000 [00:31<02:21, 13.08it/s, est. speed input: 4181.56 toks/s, output: 149.12 toks/s][A
Processed prompts:   8%|▊         | 163/2000 [00:33<02:28, 12.33it/s, est. speed input: 4435.15 toks/s, output: 155.24 toks/s][A
Processed prompts:   9%|▊         | 173/2000 [00:34<02:44, 11.12it/s, est. speed input: 4575.34 toks/s, output: 158.41 toks/s][A
Processed prompts:   9%|▉         | 177/2000 [00:35<03:25,  8.88it/s, est. speed input: 4520.15 toks/s, output: 156.00 toks/s][A
Processed prompts:  10%|▉         | 193/2000 [00:37<03:04,  9.80it/s, est. speed input: 4709.04 toks/s, output: 164.30 toks/s][A
Processed prompts:  11%|█         | 211/2000 [00:38<02:43, 10.93it/s, est. speed input: 4896.51 toks/s, output: 173.71 toks/s][A
Processed prompts:  11%|█▏        | 228/2000 [00:39<02:33, 11.54it/s, est. speed input: 5054.68 toks/s, output: 181.71 toks/s][A
Processed prompts:  12%|█▏        | 244/2000 [00:41<02:32, 11.55it/s, est. speed input: 5216.60 toks/s, output: 188.04 toks/s][A
Processed prompts:  13%|█▎        | 262/2000 [00:42<02:23, 12.14it/s, est. speed input: 5553.90 toks/s, output: 195.74 toks/s][A
Processed prompts:  14%|█▍        | 280/2000 [00:43<02:16, 12.57it/s, est. speed input: 5871.24 toks/s, output: 202.98 toks/s][A
Processed prompts:  15%|█▍        | 298/2000 [00:45<02:13, 12.78it/s, est. speed input: 6165.79 toks/s, output: 209.64 toks/s][A
Processed prompts:  15%|█▌        | 304/2000 [00:46<02:42, 10.45it/s, est. speed input: 6100.41 toks/s, output: 208.03 toks/s][A
Processed prompts:  16%|█▋        | 330/2000 [00:48<02:19, 11.98it/s, est. speed input: 6267.38 toks/s, output: 217.65 toks/s][A
Processed prompts:  18%|█▊        | 359/2000 [00:49<01:50, 14.86it/s, est. speed input: 6608.54 toks/s, output: 230.37 toks/s][A
Processed prompts:  19%|█▉        | 389/2000 [00:50<01:35, 16.90it/s, est. speed input: 6964.08 toks/s, output: 242.60 toks/s][A
Processed prompts:  20%|██        | 410/2000 [00:52<01:35, 16.64it/s, est. speed input: 7201.06 toks/s, output: 249.03 toks/s][A
Processed prompts:  21%|██        | 417/2000 [00:53<01:58, 13.33it/s, est. speed input: 7173.61 toks/s, output: 246.93 toks/s][A
Processed prompts:  22%|██▏       | 436/2000 [00:54<01:55, 13.54it/s, est. speed input: 7250.11 toks/s, output: 251.64 toks/s][A
Processed prompts:  23%|██▎       | 453/2000 [00:56<01:55, 13.39it/s, est. speed input: 7278.59 toks/s, output: 255.17 toks/s][A
Processed prompts:  24%|██▎       | 471/2000 [00:57<01:53, 13.51it/s, est. speed input: 7317.33 toks/s, output: 259.06 toks/s][A
Processed prompts:  24%|██▍       | 484/2000 [00:58<02:01, 12.47it/s, est. speed input: 7298.96 toks/s, output: 260.17 toks/s][A
Processed prompts:  24%|██▍       | 490/2000 [01:00<02:30, 10.04it/s, est. speed input: 7200.17 toks/s, output: 257.53 toks/s][A
Processed prompts:  25%|██▌       | 501/2000 [01:02<03:05,  8.06it/s, est. speed input: 7066.30 toks/s, output: 254.36 toks/s][A
Processed prompts:  25%|██▌       | 503/2000 [01:02<03:11,  7.81it/s, est. speed input: 7044.46 toks/s, output: 253.82 toks/s][A
Processed prompts:  25%|██▌       | 508/2000 [01:03<03:06,  8.01it/s, est. speed input: 7033.30 toks/s, output: 254.01 toks/s][A
Processed prompts:  26%|██▌       | 521/2000 [01:04<02:38,  9.35it/s, est. speed input: 7047.27 toks/s, output: 256.32 toks/s][A
Processed prompts:  27%|██▋       | 536/2000 [01:05<02:24, 10.16it/s, est. speed input: 7054.95 toks/s, output: 258.58 toks/s][A
Processed prompts:  28%|██▊       | 552/2000 [01:06<02:11, 11.00it/s, est. speed input: 7098.55 toks/s, output: 261.35 toks/s][A
Processed prompts:  29%|██▉       | 581/2000 [01:08<01:37, 14.62it/s, est. speed input: 7285.81 toks/s, output: 269.72 toks/s][A
Processed prompts:  30%|██▉       | 594/2000 [01:09<01:49, 12.88it/s, est. speed input: 7287.60 toks/s, output: 270.18 toks/s][A
Processed prompts:  31%|███       | 616/2000 [01:10<01:38, 14.08it/s, est. speed input: 7454.14 toks/s, output: 274.86 toks/s][A
Processed prompts:  31%|███▏      | 628/2000 [01:12<01:49, 12.56it/s, est. speed input: 7494.68 toks/s, output: 275.06 toks/s][A
Processed prompts:  32%|███▏      | 640/2000 [01:13<01:58, 11.50it/s, est. speed input: 7534.13 toks/s, output: 275.31 toks/s][A
Processed prompts:  33%|███▎      | 663/2000 [01:14<01:41, 13.20it/s, est. speed input: 7678.51 toks/s, output: 280.01 toks/s][A
Processed prompts:  34%|███▍      | 685/2000 [01:16<01:32, 14.27it/s, est. speed input: 7754.83 toks/s, output: 284.24 toks/s][A
Processed prompts:  35%|███▌      | 703/2000 [01:17<01:31, 14.10it/s, est. speed input: 7768.35 toks/s, output: 286.62 toks/s][A
Processed prompts:  36%|███▌      | 720/2000 [01:18<01:33, 13.64it/s, est. speed input: 7769.90 toks/s, output: 288.40 toks/s][A
Processed prompts:  37%|███▋      | 736/2000 [01:20<01:35, 13.20it/s, est. speed input: 7866.76 toks/s, output: 289.86 toks/s][A
Processed prompts:  38%|███▊      | 752/2000 [01:21<01:36, 12.91it/s, est. speed input: 7960.60 toks/s, output: 291.31 toks/s][A
Processed prompts:  38%|███▊      | 770/2000 [01:22<01:33, 13.15it/s, est. speed input: 8072.54 toks/s, output: 293.43 toks/s][A
Processed prompts:  39%|███▉      | 787/2000 [01:24<01:33, 12.97it/s, est. speed input: 8134.18 toks/s, output: 294.91 toks/s][A
Processed prompts:  40%|███▉      | 795/2000 [01:25<01:50, 10.91it/s, est. speed input: 8067.24 toks/s, output: 293.20 toks/s][A
Processed prompts:  40%|████      | 807/2000 [01:26<01:55, 10.37it/s, est. speed input: 8031.53 toks/s, output: 293.09 toks/s][A
Processed prompts:  41%|████      | 819/2000 [01:27<01:50, 10.70it/s, est. speed input: 8022.41 toks/s, output: 294.17 toks/s][A
Processed prompts:  41%|████▏     | 828/2000 [01:28<01:50, 10.65it/s, est. speed input: 8007.93 toks/s, output: 294.65 toks/s][A
Processed prompts:  42%|████▏     | 838/2000 [01:29<01:46, 10.87it/s, est. speed input: 8000.25 toks/s, output: 295.49 toks/s][A
Processed prompts:  43%|████▎     | 854/2000 [01:30<01:39, 11.48it/s, est. speed input: 8084.61 toks/s, output: 297.19 toks/s][A
Processed prompts:  44%|████▎     | 874/2000 [01:31<01:28, 12.74it/s, est. speed input: 8210.00 toks/s, output: 300.06 toks/s][A
Processed prompts:  45%|████▌     | 907/2000 [01:33<01:05, 16.63it/s, est. speed input: 8432.62 toks/s, output: 307.09 toks/s][A
Processed prompts:  47%|████▋     | 936/2000 [01:34<00:57, 18.43it/s, est. speed input: 8588.16 toks/s, output: 312.68 toks/s][A
Processed prompts:  47%|████▋     | 948/2000 [01:35<01:07, 15.57it/s, est. speed input: 8559.72 toks/s, output: 312.35 toks/s][A
Processed prompts:  48%|████▊     | 969/2000 [01:37<01:05, 15.74it/s, est. speed input: 8585.99 toks/s, output: 314.95 toks/s][A
Processed prompts:  50%|████▉     | 990/2000 [01:38<01:04, 15.72it/s, est. speed input: 8601.61 toks/s, output: 317.32 toks/s][A
Processed prompts:  50%|█████     | 1000/2000 [01:39<01:15, 13.29it/s, est. speed input: 8550.30 toks/s, output: 316.29 toks/s][A
Processed prompts:  50%|█████     | 1003/2000 [01:41<01:40,  9.97it/s, est. speed input: 8457.75 toks/s, output: 313.12 toks/s][A
Processed prompts:  50%|█████     | 1008/2000 [01:42<02:03,  8.02it/s, est. speed input: 8375.51 toks/s, output: 310.48 toks/s][A
Processed prompts:  51%|█████     | 1017/2000 [01:43<01:56,  8.42it/s, est. speed input: 8353.97 toks/s, output: 310.42 toks/s][A
Processed prompts:  52%|█████▏    | 1035/2000 [01:44<01:35, 10.10it/s, est. speed input: 8378.66 toks/s, output: 311.88 toks/s][A
Processed prompts:  53%|█████▎    | 1051/2000 [01:46<01:28, 10.75it/s, est. speed input: 8392.90 toks/s, output: 312.70 toks/s][A
Processed prompts:  53%|█████▎    | 1068/2000 [01:47<01:22, 11.30it/s, est. speed input: 8410.70 toks/s, output: 313.65 toks/s][A
Processed prompts:  54%|█████▍    | 1084/2000 [01:48<01:19, 11.56it/s, est. speed input: 8414.36 toks/s, output: 314.42 toks/s][A
Processed prompts:  55%|█████▌    | 1102/2000 [01:50<01:13, 12.22it/s, est. speed input: 8408.97 toks/s, output: 315.76 toks/s][A
Processed prompts:  56%|█████▌    | 1118/2000 [01:51<01:12, 12.17it/s, est. speed input: 8390.28 toks/s, output: 316.47 toks/s][A
Processed prompts:  57%|█████▋    | 1140/2000 [01:52<01:04, 13.43it/s, est. speed input: 8402.31 toks/s, output: 318.69 toks/s][A
Processed prompts:  58%|█████▊    | 1160/2000 [01:54<00:59, 14.02it/s, est. speed input: 8412.55 toks/s, output: 320.44 toks/s][A
Processed prompts:  59%|█████▉    | 1179/2000 [01:55<00:57, 14.19it/s, est. speed input: 8417.88 toks/s, output: 321.85 toks/s][A
Processed prompts:  60%|█████▉    | 1191/2000 [01:56<01:04, 12.59it/s, est. speed input: 8384.79 toks/s, output: 321.30 toks/s][A
Processed prompts:  61%|██████    | 1218/2000 [01:58<00:52, 15.02it/s, est. speed input: 8437.68 toks/s, output: 324.74 toks/s][A
Processed prompts:  62%|██████▏   | 1233/2000 [01:59<00:54, 13.99it/s, est. speed input: 8427.71 toks/s, output: 324.99 toks/s][A
Processed prompts:  62%|██████▎   | 1250/2000 [02:00<00:54, 13.69it/s, est. speed input: 8428.46 toks/s, output: 325.76 toks/s][A
Processed prompts:  63%|██████▎   | 1256/2000 [02:01<01:08, 10.84it/s, est. speed input: 8366.64 toks/s, output: 323.65 toks/s][A
Processed prompts:  63%|██████▎   | 1259/2000 [02:02<01:13, 10.08it/s, est. speed input: 8346.16 toks/s, output: 323.03 toks/s][A
Processed prompts:  63%|██████▎   | 1261/2000 [02:02<01:18,  9.47it/s, est. speed input: 8332.15 toks/s, output: 322.57 toks/s][A
Processed prompts:  64%|██████▎   | 1273/2000 [02:03<01:10, 10.37it/s, est. speed input: 8338.52 toks/s, output: 323.22 toks/s][A
Processed prompts:  65%|██████▍   | 1295/2000 [02:05<00:54, 12.82it/s, est. speed input: 8376.87 toks/s, output: 325.51 toks/s][A
Processed prompts:  66%|██████▌   | 1320/2000 [02:06<00:45, 14.86it/s, est. speed input: 8426.46 toks/s, output: 328.37 toks/s][A
Processed prompts:  67%|██████▋   | 1344/2000 [02:07<00:40, 16.02it/s, est. speed input: 8468.78 toks/s, output: 330.93 toks/s][A
Processed prompts:  68%|██████▊   | 1363/2000 [02:09<00:41, 15.54it/s, est. speed input: 8475.89 toks/s, output: 332.11 toks/s][A
Processed prompts:  69%|██████▊   | 1374/2000 [02:10<00:47, 13.27it/s, est. speed input: 8443.09 toks/s, output: 331.34 toks/s][A
Processed prompts:  69%|██████▉   | 1380/2000 [02:11<00:59, 10.43it/s, est. speed input: 8383.92 toks/s, output: 329.34 toks/s][A
Processed prompts:  69%|██████▉   | 1382/2000 [02:12<01:07,  9.21it/s, est. speed input: 8354.45 toks/s, output: 328.31 toks/s][A
Processed prompts:  70%|██████▉   | 1391/2000 [02:13<01:04,  9.45it/s, est. speed input: 8341.40 toks/s, output: 328.33 toks/s][A
Processed prompts:  70%|███████   | 1408/2000 [02:14<00:55, 10.66it/s, est. speed input: 8337.76 toks/s, output: 329.24 toks/s][A
Processed prompts:  71%|███████▏  | 1425/2000 [02:16<00:50, 11.29it/s, est. speed input: 8332.34 toks/s, output: 330.05 toks/s][A
Processed prompts:  72%|███████▏  | 1444/2000 [02:17<00:45, 12.28it/s, est. speed input: 8342.67 toks/s, output: 331.40 toks/s][A
Processed prompts:  73%|███████▎  | 1464/2000 [02:18<00:40, 13.23it/s, est. speed input: 8370.59 toks/s, output: 332.98 toks/s][A
Processed prompts:  74%|███████▍  | 1478/2000 [02:19<00:42, 12.34it/s, est. speed input: 8364.64 toks/s, output: 333.05 toks/s][A
Processed prompts:  75%|███████▌  | 1501/2000 [02:21<00:36, 13.73it/s, est. speed input: 8404.66 toks/s, output: 335.11 toks/s][A
Processed prompts:  76%|███████▌  | 1517/2000 [02:22<00:32, 14.76it/s, est. speed input: 8436.75 toks/s, output: 336.68 toks/s][A
Processed prompts:  77%|███████▋  | 1533/2000 [02:22<00:23, 19.76it/s, est. speed input: 8513.53 toks/s, output: 340.01 toks/s][A
Processed prompts:  78%|███████▊  | 1556/2000 [02:22<00:14, 29.75it/s, est. speed input: 8627.76 toks/s, output: 344.89 toks/s][A
Processed prompts:  79%|███████▉  | 1584/2000 [02:22<00:09, 45.64it/s, est. speed input: 8767.67 toks/s, output: 350.81 toks/s][A
Processed prompts:  81%|████████  | 1617/2000 [02:22<00:05, 64.97it/s, est. speed input: 8928.83 toks/s, output: 357.53 toks/s][A
Processed prompts:  82%|████████▎ | 1650/2000 [02:22<00:04, 85.80it/s, est. speed input: 9062.12 toks/s, output: 364.29 toks/s][A
Processed prompts:  84%|████████▍ | 1676/2000 [02:23<00:03, 99.08it/s, est. speed input: 9162.85 toks/s, output: 369.52 toks/s][A
Processed prompts:  85%|████████▍ | 1694/2000 [02:23<00:03, 101.94it/s, est. speed input: 9230.78 toks/s, output: 373.23 toks/s][A
Processed prompts:  86%|████████▋ | 1730/2000 [02:23<00:02, 130.15it/s, est. speed input: 9383.10 toks/s, output: 380.98 toks/s][A
Processed prompts:  88%|████████▊ | 1758/2000 [02:23<00:01, 148.43it/s, est. speed input: 9508.32 toks/s, output: 386.99 toks/s][A
Processed prompts:  89%|████████▉ | 1777/2000 [02:23<00:01, 151.59it/s, est. speed input: 9601.75 toks/s, output: 390.86 toks/s][A
Processed prompts:  90%|█████████ | 1805/2000 [02:23<00:01, 172.90it/s, est. speed input: 9743.75 toks/s, output: 396.81 toks/s][A
Processed prompts:  93%|█████████▎| 1862/2000 [02:23<00:00, 239.98it/s, est. speed input: 10044.02 toks/s, output: 409.07 toks/s][A
Processed prompts:  95%|█████████▌| 1903/2000 [02:23<00:00, 268.80it/s, est. speed input: 10248.36 toks/s, output: 417.73 toks/s][A
Processed prompts:  97%|█████████▋| 1933/2000 [02:24<00:00, 274.66it/s, est. speed input: 10396.15 toks/s, output: 424.10 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:24<00:00, 13.87it/s, est. speed input: 10735.63 toks/s, output: 438.81 toks/s] 
2025-07-02 23:57:48,032 - INFO - 
Progress: 2000 items processed
2025-07-02 23:57:48,033 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-02 23:57:48,033 - INFO -   accuracy: 0.971
2025-07-02 23:57:48,033 - INFO -   macro_f1: 0.493
2025-07-02 23:57:48,033 - INFO -   weighted_f1: 0.985
2025-07-02 23:57:48,033 - INFO - 
OVERALL metrics:
2025-07-02 23:57:48,033 - INFO -   exact_match_ratio: 0.971
2025-07-02 23:57:48,033 - INFO - 
TARGET_GROUP metrics:
2025-07-02 23:57:48,033 - INFO -   accuracy: 0.971
2025-07-02 23:57:48,033 - INFO -   macro_f1: 0.246
2025-07-02 23:57:48,033 - INFO -   weighted_f1: 0.985
2025-07-02 23:57:48,033 - INFO - 
ATTACK_METHOD metrics:
2025-07-02 23:57:48,033 - INFO -   accuracy: 0.971
2025-07-02 23:57:48,033 - INFO -   macro_f1: 0.328
2025-07-02 23:57:48,033 - INFO -   weighted_f1: 0.985
  8%|▊         | 1/12 [03:03<33:36, 183.28s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:27<15:20:36, 27.63s/it, est. speed input: 26.93 toks/s, output: 1.09 toks/s][A
Processed prompts:   0%|          | 2/2000 [00:27<6:25:09, 11.57s/it, est. speed input: 52.95 toks/s, output: 2.15 toks/s] [A
Processed prompts:   1%|          | 13/2000 [00:28<40:04,  1.21s/it, est. speed input: 334.32 toks/s, output: 13.47 toks/s][A
Processed prompts:   1%|          | 23/2000 [00:30<20:16,  1.63it/s, est. speed input: 570.94 toks/s, output: 23.05 toks/s][A
Processed prompts:   1%|▏         | 29/2000 [00:30<15:00,  2.19it/s, est. speed input: 699.24 toks/s, output: 28.56 toks/s][A
Processed prompts:   2%|▏         | 49/2000 [00:32<07:16,  4.47it/s, est. speed input: 1095.90 toks/s, output: 47.12 toks/s][A
Processed prompts:   3%|▎         | 61/2000 [00:33<05:55,  5.45it/s, est. speed input: 1277.36 toks/s, output: 56.68 toks/s][A
Processed prompts:   4%|▍         | 78/2000 [00:34<04:32,  7.06it/s, est. speed input: 1525.55 toks/s, output: 69.96 toks/s][A
Processed prompts:   5%|▍         | 97/2000 [00:36<03:39,  8.68it/s, est. speed input: 1788.85 toks/s, output: 83.87 toks/s][A
Processed prompts:   6%|▌         | 114/2000 [00:37<03:16,  9.62it/s, est. speed input: 2008.22 toks/s, output: 95.13 toks/s][A
Processed prompts:   7%|▋         | 133/2000 [00:38<02:53, 10.77it/s, est. speed input: 2248.49 toks/s, output: 107.26 toks/s][A
Processed prompts:   8%|▊         | 150/2000 [00:40<02:46, 11.13it/s, est. speed input: 2438.93 toks/s, output: 116.87 toks/s][A
Processed prompts:   9%|▉         | 180/2000 [00:41<02:08, 14.18it/s, est. speed input: 2817.58 toks/s, output: 135.66 toks/s][A
Processed prompts:  10%|▉         | 199/2000 [00:43<02:07, 14.09it/s, est. speed input: 3009.28 toks/s, output: 145.02 toks/s][A
Processed prompts:  11%|█         | 219/2000 [00:44<02:05, 14.24it/s, est. speed input: 3208.86 toks/s, output: 154.46 toks/s][A
Processed prompts:  12%|█▏        | 234/2000 [00:45<02:13, 13.18it/s, est. speed input: 3348.22 toks/s, output: 159.86 toks/s][A
Processed prompts:  13%|█▎        | 252/2000 [00:47<02:12, 13.20it/s, est. speed input: 3534.97 toks/s, output: 167.02 toks/s][A
Processed prompts:  13%|█▎        | 267/2000 [00:48<02:18, 12.54it/s, est. speed input: 3671.66 toks/s, output: 171.93 toks/s][A
Processed prompts:  14%|█▍        | 281/2000 [00:49<02:26, 11.76it/s, est. speed input: 3809.86 toks/s, output: 175.72 toks/s][A
Processed prompts:  15%|█▍        | 293/2000 [00:51<02:36, 10.91it/s, est. speed input: 3926.98 toks/s, output: 178.36 toks/s][A
Processed prompts:  15%|█▌        | 302/2000 [00:52<02:56,  9.61it/s, est. speed input: 4007.19 toks/s, output: 178.99 toks/s][A
Processed prompts:  16%|█▌        | 315/2000 [00:53<02:51,  9.84it/s, est. speed input: 4170.41 toks/s, output: 182.80 toks/s][A
Processed prompts:  16%|█▋        | 325/2000 [00:54<02:48,  9.92it/s, est. speed input: 4289.24 toks/s, output: 185.53 toks/s][A
Processed prompts:  17%|█▋        | 347/2000 [00:56<02:18, 11.89it/s, est. speed input: 4601.46 toks/s, output: 193.71 toks/s][A
Processed prompts:  18%|█▊        | 369/2000 [00:57<02:03, 13.24it/s, est. speed input: 4898.08 toks/s, output: 201.42 toks/s][A
Processed prompts:  19%|█▉        | 386/2000 [00:59<02:04, 12.99it/s, est. speed input: 5091.32 toks/s, output: 205.98 toks/s][A
Processed prompts:  20%|██        | 401/2000 [01:00<02:09, 12.32it/s, est. speed input: 5238.71 toks/s, output: 209.22 toks/s][A
Processed prompts:  21%|██        | 419/2000 [01:01<02:05, 12.62it/s, est. speed input: 5434.11 toks/s, output: 213.96 toks/s][A
Processed prompts:  22%|██▏       | 443/2000 [01:03<01:50, 14.13it/s, est. speed input: 5722.07 toks/s, output: 221.29 toks/s][A
Processed prompts:  24%|██▎       | 473/2000 [01:04<01:32, 16.52it/s, est. speed input: 6096.21 toks/s, output: 231.10 toks/s][A
Processed prompts:  25%|██▍       | 492/2000 [01:05<01:34, 15.89it/s, est. speed input: 6281.16 toks/s, output: 235.40 toks/s][A
Processed prompts:  25%|██▍       | 497/2000 [01:07<02:02, 12.29it/s, est. speed input: 6235.97 toks/s, output: 233.11 toks/s][A
Processed prompts:  25%|██▌       | 509/2000 [01:08<02:12, 11.29it/s, est. speed input: 6301.04 toks/s, output: 233.93 toks/s][A
Processed prompts:  26%|██▌       | 514/2000 [01:09<02:46,  8.93it/s, est. speed input: 6252.02 toks/s, output: 231.46 toks/s][A
Processed prompts:  26%|██▌       | 518/2000 [01:10<03:15,  7.57it/s, est. speed input: 6215.26 toks/s, output: 229.69 toks/s][A
Processed prompts:  26%|██▌       | 524/2000 [01:11<03:04,  7.98it/s, est. speed input: 6253.45 toks/s, output: 230.44 toks/s][A
Processed prompts:  26%|██▋       | 529/2000 [01:12<03:00,  8.13it/s, est. speed input: 6277.78 toks/s, output: 230.84 toks/s][A
Processed prompts:  27%|██▋       | 546/2000 [01:13<02:26,  9.91it/s, est. speed input: 6409.77 toks/s, output: 234.07 toks/s][A
Processed prompts:  28%|██▊       | 561/2000 [01:14<02:14, 10.69it/s, est. speed input: 6518.34 toks/s, output: 236.66 toks/s][A
Processed prompts:  30%|██▉       | 590/2000 [01:16<01:38, 14.37it/s, est. speed input: 6760.19 toks/s, output: 244.45 toks/s][A
Processed prompts:  31%|███       | 611/2000 [01:17<01:33, 14.88it/s, est. speed input: 6868.15 toks/s, output: 248.74 toks/s][A
Processed prompts:  31%|███▏      | 628/2000 [01:18<01:36, 14.25it/s, est. speed input: 6922.72 toks/s, output: 251.28 toks/s][A
Processed prompts:  32%|███▏      | 641/2000 [01:20<01:47, 12.69it/s, est. speed input: 6930.52 toks/s, output: 251.98 toks/s][A
Processed prompts:  32%|███▏      | 645/2000 [01:21<02:19,  9.73it/s, est. speed input: 6856.05 toks/s, output: 249.45 toks/s][A
Processed prompts:  33%|███▎      | 662/2000 [01:22<02:05, 10.64it/s, est. speed input: 6906.01 toks/s, output: 251.98 toks/s][A
Processed prompts:  34%|███▍      | 678/2000 [01:23<01:57, 11.23it/s, est. speed input: 6951.31 toks/s, output: 254.30 toks/s][A
Processed prompts:  35%|███▍      | 695/2000 [01:25<01:52, 11.59it/s, est. speed input: 6996.24 toks/s, output: 256.60 toks/s][A
Processed prompts:  35%|███▌      | 700/2000 [01:25<01:58, 11.00it/s, est. speed input: 6991.13 toks/s, output: 256.62 toks/s][A
Processed prompts:  36%|███▌      | 715/2000 [01:27<01:52, 11.39it/s, est. speed input: 7003.23 toks/s, output: 258.66 toks/s][A
Processed prompts:  37%|███▋      | 732/2000 [01:28<01:46, 11.86it/s, est. speed input: 7020.04 toks/s, output: 261.11 toks/s][A
Processed prompts:  38%|███▊      | 752/2000 [01:29<01:38, 12.72it/s, est. speed input: 7053.99 toks/s, output: 264.38 toks/s][A
Processed prompts:  39%|███▉      | 782/2000 [01:31<01:17, 15.77it/s, est. speed input: 7160.57 toks/s, output: 271.17 toks/s][A
Processed prompts:  40%|███▉      | 799/2000 [01:32<01:21, 14.80it/s, est. speed input: 7173.67 toks/s, output: 273.11 toks/s][A
Processed prompts:  41%|████      | 817/2000 [01:33<01:22, 14.38it/s, est. speed input: 7196.04 toks/s, output: 275.30 toks/s][A
Processed prompts:  42%|████▏     | 832/2000 [01:35<01:27, 13.30it/s, est. speed input: 7203.52 toks/s, output: 276.25 toks/s][A
Processed prompts:  42%|████▏     | 849/2000 [01:36<01:27, 13.14it/s, est. speed input: 7235.16 toks/s, output: 278.01 toks/s][A
Processed prompts:  43%|████▎     | 861/2000 [01:37<01:35, 11.88it/s, est. speed input: 7227.43 toks/s, output: 278.14 toks/s][A
Processed prompts:  44%|████▍     | 875/2000 [01:39<01:39, 11.34it/s, est. speed input: 7231.94 toks/s, output: 278.78 toks/s][A
Processed prompts:  45%|████▍     | 892/2000 [01:40<01:34, 11.74it/s, est. speed input: 7248.94 toks/s, output: 280.47 toks/s][A
Processed prompts:  45%|████▌     | 909/2000 [01:42<01:30, 12.00it/s, est. speed input: 7259.18 toks/s, output: 282.09 toks/s][A
Processed prompts:  46%|████▋     | 930/2000 [01:43<01:21, 13.06it/s, est. speed input: 7293.36 toks/s, output: 284.87 toks/s][A
Processed prompts:  48%|████▊     | 959/2000 [01:44<01:07, 15.35it/s, est. speed input: 7368.82 toks/s, output: 289.72 toks/s][A
Processed prompts:  49%|████▉     | 976/2000 [01:46<01:10, 14.49it/s, est. speed input: 7374.06 toks/s, output: 290.98 toks/s][A
Processed prompts:  50%|████▉     | 992/2000 [01:47<01:13, 13.67it/s, est. speed input: 7373.75 toks/s, output: 291.93 toks/s][A
Processed prompts:  50%|████▉     | 997/2000 [01:48<01:35, 10.51it/s, est. speed input: 7305.45 toks/s, output: 289.54 toks/s][A
Processed prompts:  51%|█████     | 1012/2000 [01:50<01:32, 10.65it/s, est. speed input: 7313.26 toks/s, output: 290.31 toks/s][A
Processed prompts:  51%|█████     | 1018/2000 [01:51<01:51,  8.78it/s, est. speed input: 7263.22 toks/s, output: 288.47 toks/s][A
Processed prompts:  51%|█████     | 1024/2000 [01:52<01:50,  8.81it/s, est. speed input: 7259.51 toks/s, output: 288.46 toks/s][A
Processed prompts:  51%|█████▏    | 1029/2000 [01:52<01:50,  8.80it/s, est. speed input: 7255.78 toks/s, output: 288.41 toks/s][A
Processed prompts:  52%|█████▏    | 1045/2000 [01:54<01:38,  9.69it/s, est. speed input: 7268.60 toks/s, output: 289.26 toks/s][A
Processed prompts:  54%|█████▍    | 1077/2000 [01:55<01:04, 14.40it/s, est. speed input: 7382.63 toks/s, output: 294.54 toks/s][A
Processed prompts:  55%|█████▍    | 1094/2000 [01:57<01:06, 13.71it/s, est. speed input: 7397.88 toks/s, output: 295.58 toks/s][A
Processed prompts:  56%|█████▌    | 1119/2000 [01:58<00:58, 15.15it/s, est. speed input: 7480.43 toks/s, output: 298.63 toks/s][A
Processed prompts:  57%|█████▋    | 1142/2000 [01:59<00:55, 15.56it/s, est. speed input: 7569.67 toks/s, output: 300.97 toks/s][A
Processed prompts:  58%|█████▊    | 1157/2000 [02:01<00:59, 14.19it/s, est. speed input: 7613.20 toks/s, output: 301.34 toks/s][A
Processed prompts:  58%|█████▊    | 1170/2000 [02:02<01:04, 12.78it/s, est. speed input: 7638.47 toks/s, output: 301.22 toks/s][A
Processed prompts:  59%|█████▉    | 1179/2000 [02:04<01:15, 10.81it/s, est. speed input: 7617.86 toks/s, output: 300.03 toks/s][A
Processed prompts:  60%|█████▉    | 1192/2000 [02:05<01:17, 10.42it/s, est. speed input: 7623.90 toks/s, output: 300.00 toks/s][A
Processed prompts:  60%|██████    | 1204/2000 [02:06<01:20,  9.93it/s, est. speed input: 7611.86 toks/s, output: 299.69 toks/s][A
Processed prompts:  61%|██████    | 1214/2000 [02:08<01:26,  9.14it/s, est. speed input: 7588.65 toks/s, output: 298.87 toks/s][A
Processed prompts:  61%|██████▏   | 1225/2000 [02:09<01:24,  9.12it/s, est. speed input: 7580.71 toks/s, output: 298.76 toks/s][A
Processed prompts:  62%|██████▏   | 1236/2000 [02:10<01:19,  9.61it/s, est. speed input: 7586.08 toks/s, output: 299.16 toks/s][A
Processed prompts:  62%|██████▏   | 1248/2000 [02:11<01:14, 10.13it/s, est. speed input: 7588.81 toks/s, output: 299.72 toks/s][A
Processed prompts:  63%|██████▎   | 1265/2000 [02:12<01:07, 10.88it/s, est. speed input: 7593.08 toks/s, output: 300.71 toks/s][A
Processed prompts:  64%|██████▍   | 1285/2000 [02:14<01:00, 11.89it/s, est. speed input: 7608.54 toks/s, output: 302.22 toks/s][A
Processed prompts:  66%|██████▌   | 1315/2000 [02:15<00:45, 14.97it/s, est. speed input: 7672.15 toks/s, output: 306.10 toks/s][A
Processed prompts:  67%|██████▋   | 1333/2000 [02:16<00:46, 14.36it/s, est. speed input: 7677.19 toks/s, output: 307.12 toks/s][A
Processed prompts:  68%|██████▊   | 1351/2000 [02:18<00:46, 13.94it/s, est. speed input: 7681.62 toks/s, output: 308.09 toks/s][A
Processed prompts:  68%|██████▊   | 1367/2000 [02:19<00:48, 13.13it/s, est. speed input: 7681.47 toks/s, output: 308.53 toks/s][A
Processed prompts:  69%|██████▉   | 1382/2000 [02:21<00:49, 12.48it/s, est. speed input: 7686.06 toks/s, output: 308.84 toks/s][A
Processed prompts:  70%|███████   | 1402/2000 [02:22<00:45, 13.10it/s, est. speed input: 7716.24 toks/s, output: 310.23 toks/s][A
Processed prompts:  71%|███████   | 1415/2000 [02:23<00:48, 12.01it/s, est. speed input: 7710.02 toks/s, output: 310.08 toks/s][A
Processed prompts:  71%|███████   | 1421/2000 [02:25<01:00,  9.62it/s, est. speed input: 7668.56 toks/s, output: 308.36 toks/s][A
Processed prompts:  72%|███████▏  | 1434/2000 [02:26<00:59,  9.57it/s, est. speed input: 7703.85 toks/s, output: 308.30 toks/s][A
Processed prompts:  73%|███████▎  | 1451/2000 [02:28<00:52, 10.38it/s, est. speed input: 7770.48 toks/s, output: 309.09 toks/s][A
Processed prompts:  73%|███████▎  | 1468/2000 [02:29<00:48, 10.91it/s, est. speed input: 7835.10 toks/s, output: 309.83 toks/s][A
Processed prompts:  74%|███████▍  | 1483/2000 [02:30<00:46, 11.09it/s, est. speed input: 7886.80 toks/s, output: 310.34 toks/s][A
Processed prompts:  75%|███████▌  | 1501/2000 [02:32<00:42, 11.74it/s, est. speed input: 7952.52 toks/s, output: 311.36 toks/s][A
Processed prompts:  76%|███████▌  | 1518/2000 [02:32<00:30, 15.58it/s, est. speed input: 8064.76 toks/s, output: 314.30 toks/s][A
Processed prompts:  76%|███████▌  | 1524/2000 [02:32<00:27, 17.25it/s, est. speed input: 8104.26 toks/s, output: 315.33 toks/s][A
Processed prompts:  77%|███████▋  | 1532/2000 [02:32<00:22, 20.38it/s, est. speed input: 8156.60 toks/s, output: 316.74 toks/s][A
Processed prompts:  78%|███████▊  | 1561/2000 [02:32<00:11, 39.07it/s, est. speed input: 8359.29 toks/s, output: 322.48 toks/s][A
Processed prompts:  79%|███████▉  | 1578/2000 [02:32<00:08, 50.74it/s, est. speed input: 8471.90 toks/s, output: 325.71 toks/s][A
Processed prompts:  81%|████████▏ | 1625/2000 [02:33<00:04, 89.84it/s, est. speed input: 8736.55 toks/s, output: 334.67 toks/s][A
Processed prompts:  83%|████████▎ | 1656/2000 [02:33<00:03, 108.84it/s, est. speed input: 8877.52 toks/s, output: 340.44 toks/s][A
Processed prompts:  84%|████████▎ | 1674/2000 [02:33<00:03, 108.10it/s, est. speed input: 8955.00 toks/s, output: 343.81 toks/s][A
Processed prompts:  85%|████████▍ | 1696/2000 [02:33<00:02, 115.35it/s, est. speed input: 9052.17 toks/s, output: 348.05 toks/s][A
Processed prompts:  86%|████████▌ | 1724/2000 [02:33<00:02, 131.12it/s, est. speed input: 9186.34 toks/s, output: 353.56 toks/s][A
Processed prompts:  88%|████████▊ | 1750/2000 [02:33<00:01, 142.76it/s, est. speed input: 9352.58 toks/s, output: 358.63 toks/s][A
Processed prompts:  90%|████████▉ | 1790/2000 [02:33<00:01, 187.51it/s, est. speed input: 9617.30 toks/s, output: 366.51 toks/s][A
Processed prompts:  92%|█████████▏| 1841/2000 [02:34<00:00, 233.81it/s, est. speed input: 9939.88 toks/s, output: 376.56 toks/s][A
Processed prompts:  96%|█████████▌| 1910/2000 [02:34<00:00, 322.26it/s, est. speed input: 10236.42 toks/s, output: 390.13 toks/s][A
Processed prompts:  98%|█████████▊| 1957/2000 [02:34<00:00, 336.18it/s, est. speed input: 10450.91 toks/s, output: 399.23 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:34<00:00, 12.96it/s, est. speed input: 10627.60 toks/s, output: 407.81 toks/s] 
2025-07-03 00:00:31,692 - INFO - 
Progress: 4000 items processed
2025-07-03 00:00:31,692 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:00:31,692 - INFO -   accuracy: 0.978
2025-07-03 00:00:31,692 - INFO -   macro_f1: 0.494
2025-07-03 00:00:31,692 - INFO -   weighted_f1: 0.989
2025-07-03 00:00:31,692 - INFO - 
OVERALL metrics:
2025-07-03 00:00:31,692 - INFO -   exact_match_ratio: 0.978
2025-07-03 00:00:31,692 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:00:31,692 - INFO -   accuracy: 0.978
2025-07-03 00:00:31,692 - INFO -   macro_f1: 0.247
2025-07-03 00:00:31,692 - INFO -   weighted_f1: 0.989
2025-07-03 00:00:31,692 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:00:31,692 - INFO -   accuracy: 0.978
2025-07-03 00:00:31,692 - INFO -   macro_f1: 0.330
2025-07-03 00:00:31,692 - INFO -   weighted_f1: 0.989
 17%|█▋        | 2/12 [05:46<28:37, 171.74s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:27<15:13:55, 27.43s/it, est. speed input: 22.75 toks/s, output: 1.13 toks/s][A
Processed prompts:   0%|          | 2/2000 [00:27<6:26:06, 11.60s/it, est. speed input: 44.67 toks/s, output: 2.22 toks/s] [A
Processed prompts:   1%|          | 14/2000 [00:29<37:18,  1.13s/it, est. speed input: 302.04 toks/s, output: 15.00 toks/s][A
Processed prompts:   1%|          | 16/2000 [00:29<31:49,  1.04it/s, est. speed input: 340.13 toks/s, output: 16.97 toks/s][A
Processed prompts:   2%|▏         | 34/2000 [00:30<11:20,  2.89it/s, est. speed input: 915.78 toks/s, output: 35.06 toks/s][A
Processed prompts:   3%|▎         | 51/2000 [00:32<07:05,  4.58it/s, est. speed input: 1467.53 toks/s, output: 50.51 toks/s][A
Processed prompts:   4%|▍         | 78/2000 [00:33<04:12,  7.60it/s, est. speed input: 2208.41 toks/s, output: 73.90 toks/s][A
Processed prompts:   5%|▌         | 104/2000 [00:34<03:06, 10.16it/s, est. speed input: 2781.78 toks/s, output: 94.43 toks/s][A
Processed prompts:   6%|▋         | 125/2000 [00:36<02:44, 11.38it/s, est. speed input: 3115.83 toks/s, output: 108.82 toks/s][A
Processed prompts:   7%|▋         | 140/2000 [00:37<02:45, 11.27it/s, est. speed input: 3298.44 toks/s, output: 117.21 toks/s][A
Processed prompts:   7%|▋         | 142/2000 [00:39<03:38,  8.52it/s, est. speed input: 3215.58 toks/s, output: 114.54 toks/s][A
Processed prompts:   8%|▊         | 160/2000 [00:40<03:08,  9.74it/s, est. speed input: 3434.87 toks/s, output: 124.80 toks/s][A
Processed prompts:   9%|▉         | 177/2000 [00:41<02:54, 10.43it/s, est. speed input: 3621.27 toks/s, output: 133.60 toks/s][A
Processed prompts:  10%|▉         | 192/2000 [00:43<02:53, 10.42it/s, est. speed input: 3758.65 toks/s, output: 140.25 toks/s][A
Processed prompts:  10%|▉         | 198/2000 [00:44<03:24,  8.83it/s, est. speed input: 3749.61 toks/s, output: 140.55 toks/s][A
Processed prompts:  11%|█         | 213/2000 [00:45<03:05,  9.62it/s, est. speed input: 3888.57 toks/s, output: 147.34 toks/s][A
Processed prompts:  12%|█▏        | 230/2000 [00:47<03:07,  9.43it/s, est. speed input: 3999.87 toks/s, output: 153.30 toks/s][A
Processed prompts:  12%|█▏        | 247/2000 [00:49<02:53, 10.09it/s, est. speed input: 4139.10 toks/s, output: 160.17 toks/s][A
Processed prompts:  13%|█▎        | 265/2000 [00:50<02:39, 10.89it/s, est. speed input: 4289.07 toks/s, output: 167.45 toks/s][A
Processed prompts:  14%|█▍        | 281/2000 [00:52<02:35, 11.04it/s, est. speed input: 4401.97 toks/s, output: 173.07 toks/s][A
Processed prompts:  15%|█▍        | 298/2000 [00:53<02:30, 11.34it/s, est. speed input: 4522.30 toks/s, output: 179.00 toks/s][A
Processed prompts:  16%|█▋        | 327/2000 [00:54<01:59, 14.05it/s, est. speed input: 4881.31 toks/s, output: 191.60 toks/s][A
Processed prompts:  17%|█▋        | 349/2000 [00:56<01:52, 14.63it/s, est. speed input: 5182.44 toks/s, output: 199.52 toks/s][A
Processed prompts:  18%|█▊        | 370/2000 [00:57<01:50, 14.81it/s, est. speed input: 5473.40 toks/s, output: 206.33 toks/s][A
Processed prompts:  20%|██        | 403/2000 [00:59<01:31, 17.44it/s, est. speed input: 5957.13 toks/s, output: 218.93 toks/s][A
Processed prompts:  21%|██        | 419/2000 [01:00<01:40, 15.76it/s, est. speed input: 6104.18 toks/s, output: 222.17 toks/s][A
Processed prompts:  22%|██▏       | 434/2000 [01:01<01:49, 14.33it/s, est. speed input: 6227.38 toks/s, output: 224.76 toks/s][A
Processed prompts:  22%|██▎       | 450/2000 [01:03<01:54, 13.54it/s, est. speed input: 6358.15 toks/s, output: 227.74 toks/s][A
Processed prompts:  23%|██▎       | 466/2000 [01:04<01:58, 12.91it/s, est. speed input: 6476.15 toks/s, output: 230.51 toks/s][A
Processed prompts:  24%|██▍       | 484/2000 [01:05<01:56, 13.01it/s, est. speed input: 6625.28 toks/s, output: 234.25 toks/s][A
Processed prompts:  25%|██▍       | 499/2000 [01:07<02:01, 12.39it/s, est. speed input: 6721.02 toks/s, output: 236.40 toks/s][A
Processed prompts:  25%|██▌       | 501/2000 [01:08<02:47,  8.97it/s, est. speed input: 6611.48 toks/s, output: 232.35 toks/s][A
Processed prompts:  25%|██▌       | 502/2000 [01:10<03:50,  6.50it/s, est. speed input: 6497.03 toks/s, output: 228.19 toks/s][A
Processed prompts:  26%|██▌       | 514/2000 [01:11<03:23,  7.31it/s, est. speed input: 6573.94 toks/s, output: 229.30 toks/s][A
Processed prompts:  26%|██▌       | 516/2000 [01:11<03:30,  7.05it/s, est. speed input: 6568.99 toks/s, output: 228.88 toks/s][A
Processed prompts:  27%|██▋       | 533/2000 [01:13<02:45,  8.87it/s, est. speed input: 6710.12 toks/s, output: 231.69 toks/s][A
Processed prompts:  28%|██▊       | 550/2000 [01:14<02:27,  9.86it/s, est. speed input: 6842.07 toks/s, output: 234.25 toks/s][A
Processed prompts:  28%|██▊       | 563/2000 [01:15<02:22, 10.08it/s, est. speed input: 6912.77 toks/s, output: 235.77 toks/s][A
Processed prompts:  28%|██▊       | 565/2000 [01:16<02:31,  9.47it/s, est. speed input: 6896.01 toks/s, output: 235.37 toks/s][A
Processed prompts:  29%|██▊       | 572/2000 [01:17<02:32,  9.34it/s, est. speed input: 6893.76 toks/s, output: 235.89 toks/s][A
Processed prompts:  30%|██▉       | 593/2000 [01:18<02:01, 11.58it/s, est. speed input: 6970.19 toks/s, output: 240.35 toks/s][A
Processed prompts:  30%|███       | 610/2000 [01:19<01:59, 11.61it/s, est. speed input: 7001.53 toks/s, output: 242.93 toks/s][A
Processed prompts:  32%|███▏      | 632/2000 [01:21<01:45, 13.00it/s, est. speed input: 7130.17 toks/s, output: 247.52 toks/s][A
Processed prompts:  32%|███▎      | 650/2000 [01:22<01:44, 12.96it/s, est. speed input: 7240.89 toks/s, output: 250.31 toks/s][A
Processed prompts:  33%|███▎      | 668/2000 [01:24<01:42, 12.94it/s, est. speed input: 7348.63 toks/s, output: 252.99 toks/s][A
Processed prompts:  35%|███▍      | 698/2000 [01:25<01:24, 15.44it/s, est. speed input: 7597.50 toks/s, output: 259.74 toks/s][A
Processed prompts:  36%|███▌      | 713/2000 [01:26<01:31, 14.05it/s, est. speed input: 7660.33 toks/s, output: 260.97 toks/s][A
Processed prompts:  36%|███▋      | 729/2000 [01:28<01:35, 13.29it/s, est. speed input: 7732.99 toks/s, output: 262.49 toks/s][A
Processed prompts:  37%|███▋      | 743/2000 [01:29<01:43, 12.18it/s, est. speed input: 7774.82 toks/s, output: 263.10 toks/s][A
Processed prompts:  38%|███▊      | 751/2000 [01:31<02:01, 10.27it/s, est. speed input: 7750.03 toks/s, output: 261.85 toks/s][A
Processed prompts:  38%|███▊      | 770/2000 [01:32<01:48, 11.31it/s, est. speed input: 7852.88 toks/s, output: 264.44 toks/s][A
Processed prompts:  39%|███▉      | 783/2000 [01:33<01:53, 10.73it/s, est. speed input: 7884.37 toks/s, output: 264.93 toks/s][A
Processed prompts:  40%|███▉      | 798/2000 [01:35<01:52, 10.66it/s, est. speed input: 7929.54 toks/s, output: 265.97 toks/s][A
Processed prompts:  41%|████      | 814/2000 [01:36<01:48, 10.93it/s, est. speed input: 7935.04 toks/s, output: 267.44 toks/s][A
Processed prompts:  42%|████▏     | 834/2000 [01:38<01:37, 11.97it/s, est. speed input: 7952.39 toks/s, output: 270.14 toks/s][A
Processed prompts:  42%|████▏     | 848/2000 [01:39<01:40, 11.41it/s, est. speed input: 7931.43 toks/s, output: 270.86 toks/s][A
Processed prompts:  43%|████▎     | 865/2000 [01:41<01:39, 11.43it/s, est. speed input: 7928.92 toks/s, output: 272.26 toks/s][A
Processed prompts:  44%|████▍     | 884/2000 [01:42<01:32, 12.06it/s, est. speed input: 7957.24 toks/s, output: 274.45 toks/s][A
Processed prompts:  45%|████▍     | 898/2000 [01:43<01:36, 11.43it/s, est. speed input: 7949.60 toks/s, output: 275.04 toks/s][A
Processed prompts:  46%|████▌     | 916/2000 [01:45<01:32, 11.70it/s, est. speed input: 7966.64 toks/s, output: 276.69 toks/s][A
Processed prompts:  47%|████▋     | 931/2000 [01:46<01:33, 11.41it/s, est. speed input: 7971.41 toks/s, output: 277.56 toks/s][A
Processed prompts:  47%|████▋     | 948/2000 [01:48<01:30, 11.60it/s, est. speed input: 7990.28 toks/s, output: 278.97 toks/s][A
Processed prompts:  49%|████▉     | 975/2000 [01:49<01:13, 13.90it/s, est. speed input: 8124.22 toks/s, output: 283.14 toks/s][A
Processed prompts:  50%|████▉     | 999/2000 [01:50<01:07, 14.81it/s, est. speed input: 8235.79 toks/s, output: 286.29 toks/s][A
Processed prompts:  50%|█████     | 1006/2000 [01:52<01:23, 11.91it/s, est. speed input: 8214.51 toks/s, output: 284.70 toks/s][A
Processed prompts:  51%|█████     | 1021/2000 [01:53<01:24, 11.57it/s, est. speed input: 8285.03 toks/s, output: 285.32 toks/s][A
Processed prompts:  52%|█████▏    | 1033/2000 [01:55<01:31, 10.56it/s, est. speed input: 8303.84 toks/s, output: 284.94 toks/s][A
Processed prompts:  52%|█████▏    | 1046/2000 [01:56<01:33, 10.19it/s, est. speed input: 8326.58 toks/s, output: 284.91 toks/s][A
Processed prompts:  53%|█████▎    | 1057/2000 [01:57<01:37,  9.67it/s, est. speed input: 8348.68 toks/s, output: 284.73 toks/s][A
Processed prompts:  54%|█████▎    | 1070/2000 [01:58<01:31, 10.15it/s, est. speed input: 8391.77 toks/s, output: 285.46 toks/s][A
Processed prompts:  54%|█████▍    | 1085/2000 [02:00<01:28, 10.35it/s, est. speed input: 8427.66 toks/s, output: 285.92 toks/s][A
Processed prompts:  55%|█████▍    | 1093/2000 [02:01<01:27, 10.32it/s, est. speed input: 8443.27 toks/s, output: 286.06 toks/s][A
Processed prompts:  55%|█████▍    | 1095/2000 [02:01<01:34,  9.60it/s, est. speed input: 8432.24 toks/s, output: 285.62 toks/s][A
Processed prompts:  55%|█████▌    | 1104/2000 [02:02<01:30,  9.93it/s, est. speed input: 8454.01 toks/s, output: 286.03 toks/s][A
Processed prompts:  56%|█████▌    | 1121/2000 [02:03<01:20, 10.87it/s, est. speed input: 8509.75 toks/s, output: 287.23 toks/s][A
Processed prompts:  57%|█████▋    | 1143/2000 [02:05<01:07, 12.69it/s, est. speed input: 8607.12 toks/s, output: 289.61 toks/s][A
Processed prompts:  58%|█████▊    | 1167/2000 [02:06<00:59, 14.10it/s, est. speed input: 8715.18 toks/s, output: 292.28 toks/s][A
Processed prompts:  59%|█████▉    | 1185/2000 [02:07<00:59, 13.77it/s, est. speed input: 8772.09 toks/s, output: 293.54 toks/s][A
Processed prompts:  60%|█████▉    | 1197/2000 [02:09<01:06, 12.16it/s, est. speed input: 8777.07 toks/s, output: 293.34 toks/s][A
Processed prompts:  60%|██████    | 1207/2000 [02:10<01:15, 10.52it/s, est. speed input: 8762.72 toks/s, output: 292.58 toks/s][A
Processed prompts:  61%|██████    | 1221/2000 [02:12<01:13, 10.62it/s, est. speed input: 8790.47 toks/s, output: 293.12 toks/s][A
Processed prompts:  62%|██████▏   | 1238/2000 [02:13<01:08, 11.16it/s, est. speed input: 8842.91 toks/s, output: 294.17 toks/s][A
Processed prompts:  63%|██████▎   | 1255/2000 [02:14<01:04, 11.53it/s, est. speed input: 8894.34 toks/s, output: 295.20 toks/s][A
Processed prompts:  64%|██████▎   | 1272/2000 [02:16<01:02, 11.63it/s, est. speed input: 8940.78 toks/s, output: 296.09 toks/s][A
Processed prompts:  64%|██████▍   | 1288/2000 [02:17<01:01, 11.62it/s, est. speed input: 8965.66 toks/s, output: 296.83 toks/s][A
Processed prompts:  65%|██████▌   | 1305/2000 [02:19<00:58, 11.81it/s, est. speed input: 8979.04 toks/s, output: 297.79 toks/s][A
Processed prompts:  66%|██████▋   | 1330/2000 [02:20<00:49, 13.57it/s, est. speed input: 9058.31 toks/s, output: 300.36 toks/s][A
Processed prompts:  68%|██████▊   | 1360/2000 [02:21<00:39, 16.07it/s, est. speed input: 9185.83 toks/s, output: 304.08 toks/s][A
Processed prompts:  69%|██████▉   | 1378/2000 [02:23<00:40, 15.22it/s, est. speed input: 9253.94 toks/s, output: 305.05 toks/s][A
Processed prompts:  69%|██████▉   | 1387/2000 [02:24<00:48, 12.63it/s, est. speed input: 9240.32 toks/s, output: 304.09 toks/s][A
Processed prompts:  70%|██████▉   | 1398/2000 [02:25<00:54, 11.08it/s, est. speed input: 9239.72 toks/s, output: 303.45 toks/s][A
Processed prompts:  70%|███████   | 1409/2000 [02:27<00:58, 10.12it/s, est. speed input: 9238.00 toks/s, output: 302.94 toks/s][A
Processed prompts:  71%|███████▏  | 1428/2000 [02:28<00:51, 11.18it/s, est. speed input: 9283.47 toks/s, output: 304.14 toks/s][A
Processed prompts:  73%|███████▎  | 1451/2000 [02:30<00:42, 12.80it/s, est. speed input: 9343.13 toks/s, output: 306.14 toks/s][A
Processed prompts:  74%|███████▍  | 1475/2000 [02:31<00:37, 14.08it/s, est. speed input: 9398.97 toks/s, output: 308.23 toks/s][A
Processed prompts:  75%|███████▍  | 1498/2000 [02:32<00:33, 14.90it/s, est. speed input: 9432.79 toks/s, output: 310.08 toks/s][A
Processed prompts:  75%|███████▌  | 1507/2000 [02:34<00:39, 12.41it/s, est. speed input: 9391.91 toks/s, output: 309.15 toks/s][A
Processed prompts:  76%|███████▌  | 1513/2000 [02:35<00:44, 11.02it/s, est. speed input: 9366.85 toks/s, output: 308.49 toks/s][A
Processed prompts:  77%|███████▋  | 1534/2000 [02:35<00:26, 17.51it/s, est. speed input: 9492.38 toks/s, output: 312.49 toks/s][A
Processed prompts:  78%|███████▊  | 1555/2000 [02:35<00:17, 26.09it/s, est. speed input: 9616.30 toks/s, output: 316.55 toks/s][A
Processed prompts:  78%|███████▊  | 1569/2000 [02:35<00:13, 32.84it/s, est. speed input: 9703.00 toks/s, output: 319.14 toks/s][A
Processed prompts:  79%|███████▉  | 1581/2000 [02:35<00:10, 39.64it/s, est. speed input: 9776.42 toks/s, output: 321.32 toks/s][A
Processed prompts:  80%|███████▉  | 1593/2000 [02:35<00:08, 47.70it/s, est. speed input: 9849.97 toks/s, output: 323.56 toks/s][A
Processed prompts:  80%|████████  | 1605/2000 [02:35<00:07, 50.43it/s, est. speed input: 9916.64 toks/s, output: 325.58 toks/s][A
Processed prompts:  81%|████████  | 1615/2000 [02:36<00:06, 57.18it/s, est. speed input: 9976.72 toks/s, output: 327.41 toks/s][A
Processed prompts:  82%|████████▏ | 1634/2000 [02:36<00:05, 69.82it/s, est. speed input: 10091.08 toks/s, output: 330.91 toks/s][A
Processed prompts:  84%|████████▎ | 1671/2000 [02:36<00:03, 108.42it/s, est. speed input: 10324.96 toks/s, output: 338.16 toks/s][A
Processed prompts:  85%|████████▌ | 1705/2000 [02:36<00:02, 134.78it/s, est. speed input: 10532.31 toks/s, output: 344.75 toks/s][A
Processed prompts:  87%|████████▋ | 1737/2000 [02:36<00:01, 153.67it/s, est. speed input: 10689.97 toks/s, output: 350.95 toks/s][A
Processed prompts:  88%|████████▊ | 1770/2000 [02:36<00:01, 178.80it/s, est. speed input: 10870.55 toks/s, output: 357.39 toks/s][A
Processed prompts:  90%|█████████ | 1804/2000 [02:36<00:00, 206.40it/s, est. speed input: 11092.73 toks/s, output: 364.06 toks/s][A
Processed prompts:  93%|█████████▎| 1868/2000 [02:37<00:00, 280.21it/s, est. speed input: 11429.80 toks/s, output: 376.54 toks/s][A
Processed prompts:  97%|█████████▋| 1934/2000 [02:37<00:00, 358.70it/s, est. speed input: 11823.22 toks/s, output: 389.09 toks/s][A
Processed prompts: 100%|█████████▉| 1993/2000 [02:37<00:00, 406.42it/s, est. speed input: 12059.89 toks/s, output: 400.20 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:37<00:00, 12.71it/s, est. speed input: 12085.25 toks/s, output: 401.56 toks/s] 
2025-07-03 00:03:19,500 - INFO - 
Progress: 6000 items processed
2025-07-03 00:03:19,500 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:03:19,501 - INFO -   accuracy: 0.976
2025-07-03 00:03:19,501 - INFO -   macro_f1: 0.494
2025-07-03 00:03:19,501 - INFO -   weighted_f1: 0.988
2025-07-03 00:03:19,501 - INFO - 
OVERALL metrics:
2025-07-03 00:03:19,501 - INFO -   exact_match_ratio: 0.976
2025-07-03 00:03:19,501 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:03:19,501 - INFO -   accuracy: 0.976
2025-07-03 00:03:19,501 - INFO -   macro_f1: 0.198
2025-07-03 00:03:19,501 - INFO -   weighted_f1: 0.988
2025-07-03 00:03:19,501 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:03:19,501 - INFO -   accuracy: 0.976
2025-07-03 00:03:19,501 - INFO -   macro_f1: 0.329
2025-07-03 00:03:19,501 - INFO -   weighted_f1: 0.988
 25%|██▌       | 3/12 [08:34<25:29, 169.94s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:28<15:34:08, 28.04s/it, est. speed input: 37.06 toks/s, output: 1.11 toks/s][A
Processed prompts:   0%|          | 2/2000 [00:28<6:31:02, 11.74s/it, est. speed input: 73.31 toks/s, output: 2.19 toks/s] [A
Processed prompts:   1%|          | 20/2000 [00:29<26:25,  1.25it/s, est. speed input: 695.51 toks/s, output: 20.83 toks/s][A
Processed prompts:   2%|▏         | 38/2000 [00:31<12:37,  2.59it/s, est. speed input: 1263.01 toks/s, output: 37.83 toks/s][A
Processed prompts:   3%|▎         | 57/2000 [00:32<07:49,  4.14it/s, est. speed input: 1813.02 toks/s, output: 54.29 toks/s][A
Processed prompts:   3%|▎         | 68/2000 [00:33<06:35,  4.88it/s, est. speed input: 2085.34 toks/s, output: 62.49 toks/s][A
Processed prompts:   4%|▍         | 83/2000 [00:34<05:10,  6.17it/s, est. speed input: 2455.17 toks/s, output: 73.64 toks/s][A
Processed prompts:   5%|▌         | 101/2000 [00:36<04:06,  7.71it/s, est. speed input: 2875.56 toks/s, output: 86.50 toks/s][A
Processed prompts:   6%|▌         | 120/2000 [00:37<03:26,  9.09it/s, est. speed input: 3315.56 toks/s, output: 99.10 toks/s][A
Processed prompts:   7%|▋         | 141/2000 [00:39<02:54, 10.65it/s, est. speed input: 3829.01 toks/s, output: 112.37 toks/s][A
Processed prompts:   8%|▊         | 162/2000 [00:40<02:34, 11.88it/s, est. speed input: 4316.96 toks/s, output: 124.72 toks/s][A
Processed prompts:   9%|▉         | 183/2000 [00:41<02:23, 12.68it/s, est. speed input: 4743.24 toks/s, output: 135.92 toks/s][A
Processed prompts:   9%|▉         | 189/2000 [00:43<02:54, 10.36it/s, est. speed input: 4698.88 toks/s, output: 135.81 toks/s][A
Processed prompts:  10%|█         | 207/2000 [00:44<02:41, 11.12it/s, est. speed input: 4861.31 toks/s, output: 144.27 toks/s][A
Processed prompts:  11%|█▏        | 226/2000 [00:46<02:28, 11.91it/s, est. speed input: 5071.62 toks/s, output: 152.91 toks/s][A
Processed prompts:  13%|█▎        | 251/2000 [00:47<02:08, 13.66it/s, est. speed input: 5399.04 toks/s, output: 164.72 toks/s][A
Processed prompts:  14%|█▎        | 273/2000 [00:48<01:59, 14.41it/s, est. speed input: 5685.87 toks/s, output: 174.05 toks/s][A
Processed prompts:  14%|█▍        | 279/2000 [00:50<02:30, 11.45it/s, est. speed input: 5658.58 toks/s, output: 173.00 toks/s][A
Processed prompts:  15%|█▍        | 294/2000 [00:51<02:32, 11.16it/s, est. speed input: 5810.08 toks/s, output: 177.33 toks/s][A
Processed prompts:  16%|█▌        | 320/2000 [00:53<02:04, 13.46it/s, est. speed input: 6181.29 toks/s, output: 187.86 toks/s][A
Processed prompts:  17%|█▋        | 342/2000 [00:54<01:56, 14.23it/s, est. speed input: 6456.50 toks/s, output: 195.78 toks/s][A
Processed prompts:  18%|█▊        | 351/2000 [00:55<02:18, 11.93it/s, est. speed input: 6468.59 toks/s, output: 195.94 toks/s][A
Processed prompts:  18%|█▊        | 368/2000 [00:57<02:17, 11.90it/s, est. speed input: 6578.14 toks/s, output: 200.27 toks/s][A
Processed prompts:  19%|█▉        | 386/2000 [00:58<02:11, 12.24it/s, est. speed input: 6651.10 toks/s, output: 205.08 toks/s][A
Processed prompts:  20%|██        | 403/2000 [00:59<02:10, 12.27it/s, est. speed input: 6709.25 toks/s, output: 209.16 toks/s][A
Processed prompts:  21%|██        | 420/2000 [01:01<02:09, 12.18it/s, est. speed input: 6760.23 toks/s, output: 212.91 toks/s][A
Processed prompts:  21%|██        | 422/2000 [01:02<02:55,  9.00it/s, est. speed input: 6635.81 toks/s, output: 209.26 toks/s][A
Processed prompts:  22%|██▏       | 436/2000 [01:04<02:47,  9.35it/s, est. speed input: 6655.75 toks/s, output: 211.75 toks/s][A
Processed prompts:  23%|██▎       | 453/2000 [01:05<02:30, 10.25it/s, est. speed input: 6709.10 toks/s, output: 215.61 toks/s][A
Processed prompts:  24%|██▎       | 471/2000 [01:06<02:19, 10.95it/s, est. speed input: 6768.91 toks/s, output: 219.55 toks/s][A
Processed prompts:  24%|██▍       | 487/2000 [01:08<02:16, 11.12it/s, est. speed input: 6828.77 toks/s, output: 222.58 toks/s][A
Processed prompts:  25%|██▌       | 501/2000 [01:09<02:16, 10.99it/s, est. speed input: 6902.31 toks/s, output: 224.80 toks/s][A
Processed prompts:  25%|██▌       | 503/2000 [01:09<02:22, 10.53it/s, est. speed input: 6898.46 toks/s, output: 224.62 toks/s][A
Processed prompts:  25%|██▌       | 505/2000 [01:10<02:38,  9.46it/s, est. speed input: 6878.94 toks/s, output: 223.94 toks/s][A
Processed prompts:  26%|██▌       | 520/2000 [01:11<02:23, 10.34it/s, est. speed input: 6961.95 toks/s, output: 226.64 toks/s][A
Processed prompts:  27%|██▋       | 539/2000 [01:13<02:06, 11.54it/s, est. speed input: 7074.75 toks/s, output: 230.59 toks/s][A
Processed prompts:  28%|██▊       | 552/2000 [01:14<02:10, 11.06it/s, est. speed input: 7086.81 toks/s, output: 232.20 toks/s][A
Processed prompts:  29%|██▊       | 571/2000 [01:15<01:58, 12.02it/s, est. speed input: 7115.72 toks/s, output: 236.01 toks/s][A
Processed prompts:  29%|██▉       | 585/2000 [01:17<02:04, 11.34it/s, est. speed input: 7098.99 toks/s, output: 237.49 toks/s][A
Processed prompts:  30%|███       | 600/2000 [01:18<02:01, 11.55it/s, est. speed input: 7105.78 toks/s, output: 239.84 toks/s][A
Processed prompts:  30%|███       | 602/2000 [01:18<02:06, 11.02it/s, est. speed input: 7103.55 toks/s, output: 239.68 toks/s][A
Processed prompts:  31%|███       | 615/2000 [01:19<02:05, 11.04it/s, est. speed input: 7172.44 toks/s, output: 241.54 toks/s][A
Processed prompts:  32%|███▏      | 633/2000 [01:21<01:55, 11.83it/s, est. speed input: 7287.57 toks/s, output: 244.78 toks/s][A
Processed prompts:  33%|███▎      | 654/2000 [01:22<01:45, 12.76it/s, est. speed input: 7428.44 toks/s, output: 248.80 toks/s][A
Processed prompts:  34%|███▍      | 681/2000 [01:24<01:28, 14.96it/s, est. speed input: 7642.96 toks/s, output: 255.06 toks/s][A
Processed prompts:  35%|███▌      | 706/2000 [01:25<01:21, 15.96it/s, est. speed input: 7788.58 toks/s, output: 260.15 toks/s][A
Processed prompts:  37%|███▋      | 733/2000 [01:26<01:14, 17.07it/s, est. speed input: 7949.16 toks/s, output: 265.73 toks/s][A
Processed prompts:  38%|███▊      | 756/2000 [01:28<01:14, 16.74it/s, est. speed input: 8036.71 toks/s, output: 269.49 toks/s][A
Processed prompts:  39%|███▊      | 771/2000 [01:29<01:21, 15.00it/s, est. speed input: 8041.41 toks/s, output: 270.60 toks/s][A
Processed prompts:  39%|███▉      | 775/2000 [01:31<01:48, 11.33it/s, est. speed input: 7959.09 toks/s, output: 267.87 toks/s][A
Processed prompts:  40%|███▉      | 793/2000 [01:32<01:42, 11.77it/s, est. speed input: 8020.67 toks/s, output: 269.91 toks/s][A
Processed prompts:  41%|████      | 811/2000 [01:33<01:37, 12.19it/s, est. speed input: 8107.87 toks/s, output: 272.00 toks/s][A
Processed prompts:  41%|████▏     | 827/2000 [01:35<01:37, 12.04it/s, est. speed input: 8170.14 toks/s, output: 273.36 toks/s][A
Processed prompts:  42%|████▏     | 835/2000 [01:36<01:54, 10.18it/s, est. speed input: 8142.39 toks/s, output: 272.12 toks/s][A
Processed prompts:  42%|████▏     | 849/2000 [01:38<01:54, 10.07it/s, est. speed input: 8181.76 toks/s, output: 272.73 toks/s][A
Processed prompts:  43%|████▎     | 867/2000 [01:39<01:43, 10.97it/s, est. speed input: 8271.04 toks/s, output: 274.75 toks/s][A
Processed prompts:  44%|████▍     | 884/2000 [01:40<01:38, 11.37it/s, est. speed input: 8346.97 toks/s, output: 276.38 toks/s][A
Processed prompts:  45%|████▌     | 901/2000 [01:42<01:35, 11.48it/s, est. speed input: 8411.97 toks/s, output: 277.79 toks/s][A
Processed prompts:  45%|████▌     | 909/2000 [01:43<01:51,  9.77it/s, est. speed input: 8358.15 toks/s, output: 276.54 toks/s][A
Processed prompts:  46%|████▌     | 924/2000 [01:45<01:46, 10.07it/s, est. speed input: 8356.00 toks/s, output: 277.53 toks/s][A
Processed prompts:  47%|████▋     | 938/2000 [01:46<01:45, 10.06it/s, est. speed input: 8347.01 toks/s, output: 278.20 toks/s][A
Processed prompts:  48%|████▊     | 961/2000 [01:47<01:28, 11.81it/s, est. speed input: 8416.22 toks/s, output: 281.37 toks/s][A
Processed prompts:  49%|████▉     | 985/2000 [01:49<01:15, 13.48it/s, est. speed input: 8527.75 toks/s, output: 284.93 toks/s][A
Processed prompts:  50%|█████     | 1001/2000 [01:50<01:17, 12.95it/s, est. speed input: 8576.18 toks/s, output: 286.03 toks/s][A
Processed prompts:  50%|█████     | 1003/2000 [01:52<01:44,  9.52it/s, est. speed input: 8489.61 toks/s, output: 283.08 toks/s][A
Processed prompts:  50%|█████     | 1005/2000 [01:52<02:04,  8.02it/s, est. speed input: 8443.81 toks/s, output: 281.49 toks/s][A
Processed prompts:  52%|█████▏    | 1036/2000 [01:54<01:16, 12.66it/s, est. speed input: 8584.74 toks/s, output: 286.59 toks/s][A
Processed prompts:  53%|█████▎    | 1052/2000 [01:55<01:17, 12.29it/s, est. speed input: 8584.13 toks/s, output: 287.43 toks/s][A
Processed prompts:  53%|█████▎    | 1068/2000 [01:57<01:17, 12.03it/s, est. speed input: 8583.65 toks/s, output: 288.24 toks/s][A
Processed prompts:  54%|█████▍    | 1085/2000 [01:58<01:16, 11.93it/s, est. speed input: 8585.22 toks/s, output: 289.15 toks/s][A
Processed prompts:  55%|█████▌    | 1100/2000 [01:59<01:17, 11.57it/s, est. speed input: 8578.45 toks/s, output: 289.67 toks/s][A
Processed prompts:  55%|█████▌    | 1102/2000 [02:00<01:38,  9.14it/s, est. speed input: 8514.53 toks/s, output: 287.61 toks/s][A
Processed prompts:  56%|█████▌    | 1115/2000 [02:02<01:32,  9.59it/s, est. speed input: 8508.98 toks/s, output: 288.05 toks/s][A
Processed prompts:  57%|█████▋    | 1134/2000 [02:03<01:20, 10.71it/s, est. speed input: 8535.99 toks/s, output: 289.37 toks/s][A
Processed prompts:  58%|█████▊    | 1153/2000 [02:05<01:13, 11.59it/s, est. speed input: 8597.72 toks/s, output: 290.78 toks/s][A
Processed prompts:  58%|█████▊    | 1166/2000 [02:06<01:16, 10.88it/s, est. speed input: 8635.79 toks/s, output: 290.68 toks/s][A
Processed prompts:  59%|█████▉    | 1179/2000 [02:07<01:15, 10.84it/s, est. speed input: 8685.82 toks/s, output: 291.07 toks/s][A
Processed prompts:  59%|█████▉    | 1184/2000 [02:08<01:20, 10.14it/s, est. speed input: 8688.48 toks/s, output: 290.75 toks/s][A
Processed prompts:  60%|██████    | 1205/2000 [02:09<01:07, 11.81it/s, est. speed input: 8784.23 toks/s, output: 292.71 toks/s][A
Processed prompts:  62%|██████▏   | 1231/2000 [02:11<00:54, 14.12it/s, est. speed input: 8920.45 toks/s, output: 295.88 toks/s][A
Processed prompts:  62%|██████▏   | 1248/2000 [02:12<00:55, 13.53it/s, est. speed input: 8967.50 toks/s, output: 296.76 toks/s][A
Processed prompts:  63%|██████▎   | 1265/2000 [02:13<00:56, 12.98it/s, est. speed input: 8997.05 toks/s, output: 297.51 toks/s][A
Processed prompts:  64%|██████▍   | 1282/2000 [02:15<00:56, 12.80it/s, est. speed input: 8999.11 toks/s, output: 298.39 toks/s][A
Processed prompts:  65%|██████▍   | 1299/2000 [02:16<00:55, 12.66it/s, est. speed input: 9000.90 toks/s, output: 299.23 toks/s][A
Processed prompts:  66%|██████▌   | 1315/2000 [02:18<00:56, 12.18it/s, est. speed input: 8993.54 toks/s, output: 299.71 toks/s][A
Processed prompts:  66%|██████▌   | 1321/2000 [02:19<01:09,  9.82it/s, est. speed input: 8938.39 toks/s, output: 298.07 toks/s][A
Processed prompts:  67%|██████▋   | 1331/2000 [02:20<01:11,  9.37it/s, est. speed input: 8936.83 toks/s, output: 297.76 toks/s][A
Processed prompts:  67%|██████▋   | 1348/2000 [02:22<01:03, 10.27it/s, est. speed input: 8976.78 toks/s, output: 298.69 toks/s][A
Processed prompts:  68%|██████▊   | 1365/2000 [02:23<00:58, 10.77it/s, est. speed input: 9013.22 toks/s, output: 299.50 toks/s][A
Processed prompts:  69%|██████▉   | 1381/2000 [02:24<00:56, 11.03it/s, est. speed input: 9046.43 toks/s, output: 300.19 toks/s][A
Processed prompts:  70%|██████▉   | 1397/2000 [02:26<00:52, 11.39it/s, est. speed input: 9101.10 toks/s, output: 301.01 toks/s][A
Processed prompts:  71%|███████   | 1414/2000 [02:27<00:50, 11.67it/s, est. speed input: 9158.50 toks/s, output: 301.88 toks/s][A
Processed prompts:  72%|███████▏  | 1438/2000 [02:29<00:42, 13.30it/s, est. speed input: 9244.89 toks/s, output: 304.12 toks/s][A
Processed prompts:  73%|███████▎  | 1462/2000 [02:30<00:36, 14.59it/s, est. speed input: 9303.09 toks/s, output: 306.38 toks/s][A
Processed prompts:  74%|███████▍  | 1481/2000 [02:31<00:36, 14.39it/s, est. speed input: 9304.69 toks/s, output: 307.52 toks/s][A
Processed prompts:  75%|███████▍  | 1496/2000 [02:33<00:37, 13.26it/s, est. speed input: 9294.22 toks/s, output: 307.74 toks/s][A
Processed prompts:  76%|███████▌  | 1512/2000 [02:34<00:38, 12.82it/s, est. speed input: 9322.66 toks/s, output: 308.22 toks/s][A
Processed prompts:  76%|███████▋  | 1527/2000 [02:35<00:30, 15.27it/s, est. speed input: 9419.85 toks/s, output: 310.25 toks/s][A
Processed prompts:  77%|███████▋  | 1546/2000 [02:35<00:20, 21.69it/s, est. speed input: 9571.71 toks/s, output: 313.78 toks/s][A
Processed prompts:  78%|███████▊  | 1561/2000 [02:35<00:15, 28.08it/s, est. speed input: 9681.76 toks/s, output: 316.52 toks/s][A
Processed prompts:  78%|███████▊  | 1570/2000 [02:35<00:13, 32.08it/s, est. speed input: 9744.22 toks/s, output: 318.14 toks/s][A
Processed prompts:  79%|███████▉  | 1579/2000 [02:35<00:11, 36.88it/s, est. speed input: 9798.69 toks/s, output: 319.75 toks/s][A
Processed prompts:  80%|███████▉  | 1593/2000 [02:35<00:08, 48.16it/s, est. speed input: 9888.13 toks/s, output: 322.39 toks/s][A
Processed prompts:  81%|████████  | 1612/2000 [02:35<00:05, 67.32it/s, est. speed input: 10014.93 toks/s, output: 325.98 toks/s][A
Processed prompts:  83%|████████▎ | 1654/2000 [02:35<00:03, 111.43it/s, est. speed input: 10300.16 toks/s, output: 333.94 toks/s][A
Processed prompts:  84%|████████▎ | 1671/2000 [02:35<00:03, 108.52it/s, est. speed input: 10411.19 toks/s, output: 337.01 toks/s][A
Processed prompts:  85%|████████▌ | 1700/2000 [02:36<00:02, 127.09it/s, est. speed input: 10560.32 toks/s, output: 342.48 toks/s][A
Processed prompts:  87%|████████▋ | 1743/2000 [02:36<00:01, 166.93it/s, est. speed input: 10773.45 toks/s, output: 350.61 toks/s][A
Processed prompts:  89%|████████▉ | 1780/2000 [02:36<00:01, 196.87it/s, est. speed input: 11004.00 toks/s, output: 357.48 toks/s][A
Processed prompts:  90%|█████████ | 1803/2000 [02:36<00:01, 158.86it/s, est. speed input: 11143.62 toks/s, output: 361.51 toks/s][A
Processed prompts:  92%|█████████▏| 1839/2000 [02:36<00:00, 197.03it/s, est. speed input: 11379.47 toks/s, output: 368.84 toks/s][A
Processed prompts:  95%|█████████▍| 1898/2000 [02:36<00:00, 266.64it/s, est. speed input: 11722.69 toks/s, output: 380.68 toks/s][A
Processed prompts:  97%|█████████▋| 1947/2000 [02:36<00:00, 316.13it/s, est. speed input: 11963.91 toks/s, output: 390.43 toks/s][A
Processed prompts: 100%|█████████▉| 1999/2000 [02:37<00:00, 363.11it/s, est. speed input: 12207.26 toks/s, output: 400.85 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:37<00:00, 12.73it/s, est. speed input: 12212.22 toks/s, output: 401.06 toks/s] 
2025-07-03 00:06:07,454 - INFO - 
Progress: 8000 items processed
2025-07-03 00:06:07,454 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:06:07,454 - INFO -   accuracy: 0.968
2025-07-03 00:06:07,454 - INFO -   macro_f1: 0.492
2025-07-03 00:06:07,454 - INFO -   weighted_f1: 0.984
2025-07-03 00:06:07,454 - INFO - 
OVERALL metrics:
2025-07-03 00:06:07,454 - INFO -   exact_match_ratio: 0.968
2025-07-03 00:06:07,454 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:06:07,454 - INFO -   accuracy: 0.968
2025-07-03 00:06:07,454 - INFO -   macro_f1: 0.197
2025-07-03 00:06:07,454 - INFO -   weighted_f1: 0.984
2025-07-03 00:06:07,454 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:06:07,454 - INFO -   accuracy: 0.968
2025-07-03 00:06:07,455 - INFO -   macro_f1: 0.328
2025-07-03 00:06:07,455 - INFO -   weighted_f1: 0.984
 33%|███▎      | 4/12 [11:22<22:33, 169.16s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:27<15:21:06, 27.65s/it, est. speed input: 28.65 toks/s, output: 1.12 toks/s][A
Processed prompts:   0%|          | 2/2000 [00:27<6:25:29, 11.58s/it, est. speed input: 56.34 toks/s, output: 2.29 toks/s] [A
Processed prompts:   1%|          | 21/2000 [00:29<24:44,  1.33it/s, est. speed input: 562.60 toks/s, output: 22.05 toks/s][A
Processed prompts:   3%|▎         | 51/2000 [00:30<08:49,  3.68it/s, est. speed input: 1301.80 toks/s, output: 51.56 toks/s][A
Processed prompts:   4%|▍         | 85/2000 [00:32<04:47,  6.66it/s, est. speed input: 2078.36 toks/s, output: 82.95 toks/s][A
Processed prompts:   5%|▍         | 94/2000 [00:33<04:46,  6.65it/s, est. speed input: 2204.23 toks/s, output: 87.84 toks/s][A
Processed prompts:   5%|▌         | 105/2000 [00:34<04:35,  6.87it/s, est. speed input: 2417.95 toks/s, output: 94.15 toks/s][A
Processed prompts:   6%|▌         | 123/2000 [00:36<03:48,  8.22it/s, est. speed input: 2839.82 toks/s, output: 106.09 toks/s][A
Processed prompts:   7%|▋         | 142/2000 [00:37<03:14,  9.55it/s, est. speed input: 3273.67 toks/s, output: 117.96 toks/s][A
Processed prompts:   8%|▊         | 161/2000 [00:40<03:38,  8.43it/s, est. speed input: 3548.15 toks/s, output: 124.44 toks/s][A
Processed prompts:   9%|▉         | 179/2000 [00:41<03:12,  9.47it/s, est. speed input: 3890.01 toks/s, output: 134.11 toks/s][A
Processed prompts:  10%|▉         | 195/2000 [00:43<03:00,  9.99it/s, est. speed input: 4160.29 toks/s, output: 141.70 toks/s][A
Processed prompts:  11%|█         | 213/2000 [00:44<02:47, 10.68it/s, est. speed input: 4456.97 toks/s, output: 150.08 toks/s][A
Processed prompts:  12%|█▏        | 230/2000 [00:45<02:38, 11.14it/s, est. speed input: 4623.23 toks/s, output: 157.44 toks/s][A
Processed prompts:  12%|█▏        | 248/2000 [00:47<02:30, 11.66it/s, est. speed input: 4724.59 toks/s, output: 165.02 toks/s][A
Processed prompts:  13%|█▎        | 266/2000 [00:48<02:23, 12.06it/s, est. speed input: 4821.76 toks/s, output: 172.19 toks/s][A
Processed prompts:  14%|█▍        | 281/2000 [00:50<02:28, 11.57it/s, est. speed input: 4879.35 toks/s, output: 176.83 toks/s][A
Processed prompts:  15%|█▍        | 299/2000 [00:51<02:21, 12.01it/s, est. speed input: 5130.72 toks/s, output: 183.29 toks/s][A
Processed prompts:  16%|█▋        | 327/2000 [00:52<01:55, 14.43it/s, est. speed input: 5510.42 toks/s, output: 195.07 toks/s][A
Processed prompts:  18%|█▊        | 357/2000 [00:54<01:38, 16.63it/s, est. speed input: 5892.35 toks/s, output: 207.37 toks/s][A
Processed prompts:  19%|█▉        | 379/2000 [00:55<01:39, 16.27it/s, est. speed input: 6073.19 toks/s, output: 214.19 toks/s][A
Processed prompts:  20%|█▉        | 392/2000 [00:57<01:52, 14.25it/s, est. speed input: 6096.28 toks/s, output: 215.96 toks/s][A
Processed prompts:  20%|█▉        | 394/2000 [00:58<02:33, 10.44it/s, est. speed input: 5977.44 toks/s, output: 211.98 toks/s][A
Processed prompts:  20%|██        | 406/2000 [00:59<02:42,  9.79it/s, est. speed input: 5982.84 toks/s, output: 213.21 toks/s][A
Processed prompts:  21%|██        | 424/2000 [01:01<02:26, 10.76it/s, est. speed input: 6066.31 toks/s, output: 217.73 toks/s][A
Processed prompts:  22%|██▏       | 441/2000 [01:02<02:19, 11.21it/s, est. speed input: 6134.13 toks/s, output: 221.52 toks/s][A
Processed prompts:  23%|██▎       | 457/2000 [01:04<02:16, 11.30it/s, est. speed input: 6187.85 toks/s, output: 224.65 toks/s][A
Processed prompts:  24%|██▍       | 478/2000 [01:05<02:03, 12.30it/s, est. speed input: 6290.07 toks/s, output: 229.82 toks/s][A
Processed prompts:  25%|██▍       | 498/2000 [01:06<01:56, 12.92it/s, est. speed input: 6381.14 toks/s, output: 234.43 toks/s][A
Processed prompts:  25%|██▌       | 501/2000 [01:09<03:12,  7.80it/s, est. speed input: 6180.40 toks/s, output: 227.25 toks/s][A
Processed prompts:  25%|██▌       | 503/2000 [01:09<03:21,  7.41it/s, est. speed input: 6158.92 toks/s, output: 226.55 toks/s][A
Processed prompts:  26%|██▌       | 522/2000 [01:11<02:38,  9.31it/s, est. speed input: 6244.16 toks/s, output: 230.26 toks/s][A
Processed prompts:  27%|██▋       | 547/2000 [01:12<02:01, 11.93it/s, est. speed input: 6389.03 toks/s, output: 236.41 toks/s][A
Processed prompts:  28%|██▊       | 565/2000 [01:14<01:57, 12.26it/s, est. speed input: 6460.33 toks/s, output: 239.36 toks/s][A
Processed prompts:  29%|██▊       | 574/2000 [01:15<02:17, 10.38it/s, est. speed input: 6429.33 toks/s, output: 238.39 toks/s][A
Processed prompts:  29%|██▉       | 578/2000 [01:16<02:33,  9.24it/s, est. speed input: 6399.75 toks/s, output: 237.43 toks/s][A
Processed prompts:  29%|██▉       | 581/2000 [01:16<02:41,  8.77it/s, est. speed input: 6392.60 toks/s, output: 237.16 toks/s][A
Processed prompts:  29%|██▉       | 585/2000 [01:17<02:47,  8.43it/s, est. speed input: 6398.95 toks/s, output: 237.11 toks/s][A
Processed prompts:  30%|███       | 603/2000 [01:18<02:16, 10.22it/s, est. speed input: 6521.38 toks/s, output: 240.44 toks/s][A
Processed prompts:  31%|███       | 620/2000 [01:20<02:06, 10.92it/s, est. speed input: 6627.03 toks/s, output: 243.23 toks/s][A
Processed prompts:  32%|███▎      | 650/2000 [01:21<01:34, 14.35it/s, est. speed input: 6898.74 toks/s, output: 250.88 toks/s][A
Processed prompts:  34%|███▎      | 670/2000 [01:23<01:32, 14.35it/s, est. speed input: 7044.04 toks/s, output: 254.40 toks/s][A
Processed prompts:  34%|███▍      | 689/2000 [01:24<01:32, 14.11it/s, est. speed input: 7172.92 toks/s, output: 257.36 toks/s][A
Processed prompts:  36%|███▌      | 720/2000 [01:25<01:17, 16.47it/s, est. speed input: 7440.77 toks/s, output: 264.39 toks/s][A
Processed prompts:  37%|███▋      | 737/2000 [01:27<01:23, 15.21it/s, est. speed input: 7530.39 toks/s, output: 266.25 toks/s][A
Processed prompts:  38%|███▊      | 756/2000 [01:28<01:24, 14.76it/s, est. speed input: 7641.08 toks/s, output: 268.71 toks/s][A
Processed prompts:  39%|███▊      | 771/2000 [01:30<01:30, 13.56it/s, est. speed input: 7700.36 toks/s, output: 269.65 toks/s][A
Processed prompts:  39%|███▉      | 789/2000 [01:31<01:31, 13.22it/s, est. speed input: 7789.03 toks/s, output: 271.43 toks/s][A
Processed prompts:  40%|████      | 804/2000 [01:32<01:35, 12.51it/s, est. speed input: 7843.95 toks/s, output: 272.43 toks/s][A
Processed prompts:  41%|████      | 820/2000 [01:34<01:36, 12.23it/s, est. speed input: 7909.89 toks/s, output: 273.63 toks/s][A
Processed prompts:  42%|████▏     | 841/2000 [01:35<01:29, 13.01it/s, est. speed input: 8025.98 toks/s, output: 276.38 toks/s][A
Processed prompts:  43%|████▎     | 858/2000 [01:37<01:28, 12.84it/s, est. speed input: 8099.21 toks/s, output: 277.85 toks/s][A
Processed prompts:  43%|████▎     | 869/2000 [01:38<01:39, 11.40it/s, est. speed input: 8105.11 toks/s, output: 277.45 toks/s][A
Processed prompts:  44%|████▍     | 875/2000 [01:39<02:01,  9.29it/s, est. speed input: 8057.10 toks/s, output: 275.54 toks/s][A
Processed prompts:  44%|████▍     | 885/2000 [01:41<02:10,  8.57it/s, est. speed input: 8047.56 toks/s, output: 274.78 toks/s][A
Processed prompts:  45%|████▌     | 902/2000 [01:42<01:53,  9.66it/s, est. speed input: 8114.57 toks/s, output: 276.36 toks/s][A
Processed prompts:  46%|████▌     | 919/2000 [01:44<01:43, 10.42it/s, est. speed input: 8179.92 toks/s, output: 277.89 toks/s][A
Processed prompts:  47%|████▋     | 936/2000 [01:45<01:36, 10.98it/s, est. speed input: 8244.19 toks/s, output: 279.40 toks/s][A
Processed prompts:  47%|████▋     | 941/2000 [01:46<02:01,  8.70it/s, est. speed input: 8183.73 toks/s, output: 277.18 toks/s][A
Processed prompts:  48%|████▊     | 952/2000 [01:48<02:00,  8.67it/s, est. speed input: 8192.72 toks/s, output: 277.26 toks/s][A
Processed prompts:  49%|████▊     | 972/2000 [01:49<01:38, 10.43it/s, est. speed input: 8272.95 toks/s, output: 279.74 toks/s][A
Processed prompts:  50%|█████     | 1000/2000 [01:50<01:15, 13.25it/s, est. speed input: 8398.26 toks/s, output: 284.16 toks/s][A
Processed prompts:  52%|█████▏    | 1030/2000 [01:52<01:01, 15.79it/s, est. speed input: 8527.95 toks/s, output: 289.06 toks/s][A
Processed prompts:  52%|█████▏    | 1047/2000 [01:53<01:04, 14.74it/s, est. speed input: 8535.46 toks/s, output: 290.09 toks/s][A
Processed prompts:  53%|█████▎    | 1055/2000 [01:55<01:18, 12.05it/s, est. speed input: 8484.30 toks/s, output: 288.72 toks/s][A
Processed prompts:  53%|█████▎    | 1059/2000 [01:56<01:41,  9.23it/s, est. speed input: 8406.41 toks/s, output: 286.30 toks/s][A
Processed prompts:  53%|█████▎    | 1069/2000 [01:57<01:47,  8.64it/s, est. speed input: 8395.96 toks/s, output: 285.67 toks/s][A
Processed prompts:  54%|█████▍    | 1078/2000 [01:58<01:42,  8.99it/s, est. speed input: 8414.40 toks/s, output: 285.99 toks/s][A
Processed prompts:  54%|█████▍    | 1081/2000 [01:59<01:46,  8.59it/s, est. speed input: 8407.06 toks/s, output: 285.64 toks/s][A
Processed prompts:  54%|█████▍    | 1085/2000 [01:59<01:49,  8.36it/s, est. speed input: 8404.29 toks/s, output: 285.40 toks/s][A
Processed prompts:  55%|█████▌    | 1101/2000 [02:01<01:31,  9.79it/s, est. speed input: 8452.20 toks/s, output: 286.48 toks/s][A
Processed prompts:  56%|█████▌    | 1118/2000 [02:02<01:23, 10.50it/s, est. speed input: 8499.35 toks/s, output: 287.51 toks/s][A
Processed prompts:  57%|█████▋    | 1136/2000 [02:03<01:16, 11.34it/s, est. speed input: 8562.35 toks/s, output: 288.91 toks/s][A
Processed prompts:  58%|█████▊    | 1156/2000 [02:05<01:08, 12.33it/s, est. speed input: 8639.94 toks/s, output: 290.75 toks/s][A
Processed prompts:  59%|█████▉    | 1185/2000 [02:06<00:54, 14.87it/s, est. speed input: 8785.05 toks/s, output: 294.53 toks/s][A
Processed prompts:  61%|██████    | 1212/2000 [02:08<00:48, 16.30it/s, est. speed input: 8913.98 toks/s, output: 297.87 toks/s][A
Processed prompts:  61%|██████▏   | 1227/2000 [02:09<00:52, 14.64it/s, est. speed input: 8938.96 toks/s, output: 298.23 toks/s][A
Processed prompts:  62%|██████▏   | 1233/2000 [02:10<01:06, 11.50it/s, est. speed input: 8891.95 toks/s, output: 296.50 toks/s][A
Processed prompts:  62%|██████▏   | 1239/2000 [02:12<01:22,  9.24it/s, est. speed input: 8842.80 toks/s, output: 294.75 toks/s][A
Processed prompts:  63%|██████▎   | 1253/2000 [02:13<01:18,  9.52it/s, est. speed input: 8866.17 toks/s, output: 295.06 toks/s][A
Processed prompts:  64%|██████▎   | 1271/2000 [02:15<01:08, 10.57it/s, est. speed input: 8915.53 toks/s, output: 296.27 toks/s][A
Processed prompts:  65%|██████▍   | 1298/2000 [02:16<00:53, 13.11it/s, est. speed input: 9012.60 toks/s, output: 299.34 toks/s][A
Processed prompts:  66%|██████▌   | 1317/2000 [02:17<00:51, 13.34it/s, est. speed input: 9060.98 toks/s, output: 300.72 toks/s][A
Processed prompts:  67%|██████▋   | 1334/2000 [02:19<00:51, 13.02it/s, est. speed input: 9061.82 toks/s, output: 301.49 toks/s][A
Processed prompts:  67%|██████▋   | 1347/2000 [02:20<00:54, 11.94it/s, est. speed input: 9041.40 toks/s, output: 301.49 toks/s][A
Processed prompts:  68%|██████▊   | 1362/2000 [02:22<00:55, 11.52it/s, est. speed input: 9047.93 toks/s, output: 301.79 toks/s][A
Processed prompts:  69%|██████▉   | 1386/2000 [02:23<00:46, 13.27it/s, est. speed input: 9125.13 toks/s, output: 304.12 toks/s][A
Processed prompts:  70%|███████   | 1403/2000 [02:24<00:45, 13.02it/s, est. speed input: 9163.80 toks/s, output: 304.89 toks/s][A
Processed prompts:  71%|███████   | 1420/2000 [02:26<00:45, 12.75it/s, est. speed input: 9199.81 toks/s, output: 305.57 toks/s][A
Processed prompts:  71%|███████   | 1422/2000 [02:28<01:07,  8.56it/s, est. speed input: 9098.32 toks/s, output: 302.26 toks/s][A
Processed prompts:  72%|███████▏  | 1436/2000 [02:29<01:01,  9.25it/s, est. speed input: 9103.12 toks/s, output: 302.82 toks/s][A
Processed prompts:  73%|███████▎  | 1455/2000 [02:30<00:51, 10.58it/s, est. speed input: 9133.00 toks/s, output: 304.18 toks/s][A
Processed prompts:  74%|███████▍  | 1487/2000 [02:32<00:36, 14.16it/s, est. speed input: 9248.74 toks/s, output: 308.08 toks/s][A
Processed prompts:  76%|███████▌  | 1517/2000 [02:33<00:29, 16.41it/s, est. speed input: 9359.85 toks/s, output: 311.54 toks/s][A
Processed prompts:  77%|███████▋  | 1545/2000 [02:34<00:25, 18.03it/s, est. speed input: 9454.13 toks/s, output: 314.55 toks/s][A
Processed prompts:  78%|███████▊  | 1567/2000 [02:34<00:17, 24.16it/s, est. speed input: 9565.57 toks/s, output: 318.64 toks/s][A
Processed prompts:  79%|███████▉  | 1576/2000 [02:34<00:15, 26.89it/s, est. speed input: 9602.22 toks/s, output: 320.19 toks/s][A
Processed prompts:  79%|███████▉  | 1584/2000 [02:35<00:13, 29.85it/s, est. speed input: 9634.47 toks/s, output: 321.54 toks/s][A
Processed prompts:  80%|███████▉  | 1598/2000 [02:35<00:11, 35.43it/s, est. speed input: 9691.40 toks/s, output: 323.83 toks/s][A
Processed prompts:  80%|████████  | 1606/2000 [02:35<00:10, 39.32it/s, est. speed input: 9737.47 toks/s, output: 325.17 toks/s][A
Processed prompts:  82%|████████▏ | 1633/2000 [02:35<00:06, 60.16it/s, est. speed input: 9906.91 toks/s, output: 330.06 toks/s][A
Processed prompts:  83%|████████▎ | 1654/2000 [02:35<00:04, 72.25it/s, est. speed input: 10039.29 toks/s, output: 333.97 toks/s][A
Processed prompts:  85%|████████▍ | 1699/2000 [02:35<00:02, 116.87it/s, est. speed input: 10336.08 toks/s, output: 342.71 toks/s][A
Processed prompts:  87%|████████▋ | 1732/2000 [02:36<00:01, 139.78it/s, est. speed input: 10530.01 toks/s, output: 348.94 toks/s][A
Processed prompts:  88%|████████▊ | 1766/2000 [02:36<00:01, 168.00it/s, est. speed input: 10692.20 toks/s, output: 355.43 toks/s][A
Processed prompts:  89%|████████▉ | 1787/2000 [02:36<00:01, 170.54it/s, est. speed input: 10802.12 toks/s, output: 359.39 toks/s][A
Processed prompts:  91%|█████████ | 1817/2000 [02:36<00:00, 192.79it/s, est. speed input: 10998.30 toks/s, output: 365.27 toks/s][A
Processed prompts:  94%|█████████▍| 1883/2000 [02:36<00:00, 278.22it/s, est. speed input: 11349.92 toks/s, output: 378.21 toks/s][A
Processed prompts:  96%|█████████▌| 1923/2000 [02:36<00:00, 304.83it/s, est. speed input: 11513.06 toks/s, output: 385.75 toks/s][A
Processed prompts:  98%|█████████▊| 1962/2000 [02:36<00:00, 307.67it/s, est. speed input: 11670.02 toks/s, output: 393.32 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:36<00:00, 12.75it/s, est. speed input: 11915.41 toks/s, output: 401.08 toks/s] 
2025-07-03 00:08:55,123 - INFO - 
Progress: 10000 items processed
2025-07-03 00:08:55,123 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:08:55,123 - INFO -   accuracy: 0.967
2025-07-03 00:08:55,123 - INFO -   macro_f1: 0.492
2025-07-03 00:08:55,123 - INFO -   weighted_f1: 0.983
2025-07-03 00:08:55,123 - INFO - 
OVERALL metrics:
2025-07-03 00:08:55,123 - INFO -   exact_match_ratio: 0.967
2025-07-03 00:08:55,123 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:08:55,123 - INFO -   accuracy: 0.967
2025-07-03 00:08:55,123 - INFO -   macro_f1: 0.197
2025-07-03 00:08:55,123 - INFO -   weighted_f1: 0.983
2025-07-03 00:08:55,123 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:08:55,123 - INFO -   accuracy: 0.967
2025-07-03 00:08:55,123 - INFO -   macro_f1: 0.246
2025-07-03 00:08:55,123 - INFO -   weighted_f1: 0.983
 42%|████▏     | 5/12 [14:10<19:40, 168.62s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:27<15:30:36, 27.93s/it, est. speed input: 38.09 toks/s, output: 1.11 toks/s][A
Processed prompts:   0%|          | 5/2000 [00:28<2:23:04,  4.30s/it, est. speed input: 186.61 toks/s, output: 5.49 toks/s][A
Processed prompts:   1%|          | 12/2000 [00:29<48:00,  1.45s/it, est. speed input: 435.73 toks/s, output: 12.98 toks/s][A
Processed prompts:   1%|          | 19/2000 [00:30<25:59,  1.27it/s, est. speed input: 671.93 toks/s, output: 20.33 toks/s][A
Processed prompts:   2%|▏         | 39/2000 [00:31<10:04,  3.24it/s, est. speed input: 1282.41 toks/s, output: 40.24 toks/s][A
Processed prompts:   3%|▎         | 65/2000 [00:32<05:24,  5.97it/s, est. speed input: 2025.96 toks/s, output: 64.17 toks/s][A
Processed prompts:   4%|▍         | 81/2000 [00:34<04:33,  7.03it/s, est. speed input: 2422.44 toks/s, output: 76.54 toks/s][A
Processed prompts:   6%|▌         | 114/2000 [00:35<02:55, 10.76it/s, est. speed input: 3296.86 toks/s, output: 103.13 toks/s][A
Processed prompts:   7%|▋         | 132/2000 [00:37<02:45, 11.29it/s, est. speed input: 3689.91 toks/s, output: 114.55 toks/s][A
Processed prompts:   7%|▋         | 141/2000 [00:38<03:04, 10.07it/s, est. speed input: 3805.46 toks/s, output: 117.91 toks/s][A
Processed prompts:   7%|▋         | 143/2000 [00:39<04:02,  7.65it/s, est. speed input: 3719.01 toks/s, output: 115.22 toks/s][A
Processed prompts:   8%|▊         | 160/2000 [00:41<03:26,  8.90it/s, est. speed input: 4032.13 toks/s, output: 124.84 toks/s][A
Processed prompts:   9%|▉         | 177/2000 [00:42<03:05,  9.84it/s, est. speed input: 4323.80 toks/s, output: 133.86 toks/s][A
Processed prompts:  10%|█         | 200/2000 [00:44<02:34, 11.63it/s, est. speed input: 4732.49 toks/s, output: 146.47 toks/s][A
Processed prompts:  11%|█▏        | 228/2000 [00:45<02:05, 14.09it/s, est. speed input: 5221.63 toks/s, output: 161.79 toks/s][A
Processed prompts:  13%|█▎        | 263/2000 [00:46<01:40, 17.37it/s, est. speed input: 5741.32 toks/s, output: 180.19 toks/s][A
Processed prompts:  15%|█▍        | 292/2000 [00:48<01:32, 18.47it/s, est. speed input: 6108.64 toks/s, output: 193.69 toks/s][A
Processed prompts:  16%|█▌        | 310/2000 [00:49<01:41, 16.66it/s, est. speed input: 6201.96 toks/s, output: 199.05 toks/s][A
Processed prompts:  16%|█▌        | 316/2000 [00:51<02:09, 13.03it/s, est. speed input: 6120.93 toks/s, output: 197.26 toks/s][A
Processed prompts:  16%|█▌        | 318/2000 [00:52<02:55,  9.57it/s, est. speed input: 5986.76 toks/s, output: 193.25 toks/s][A
Processed prompts:  16%|█▌        | 321/2000 [00:54<03:50,  7.27it/s, est. speed input: 5873.44 toks/s, output: 189.86 toks/s][A
Processed prompts:  16%|█▋        | 330/2000 [00:55<03:57,  7.03it/s, est. speed input: 5899.27 toks/s, output: 190.43 toks/s][A
Processed prompts:  17%|█▋        | 347/2000 [00:56<03:13,  8.55it/s, est. speed input: 6071.26 toks/s, output: 195.62 toks/s][A
Processed prompts:  18%|█▊        | 365/2000 [00:58<02:46,  9.83it/s, est. speed input: 6254.14 toks/s, output: 201.11 toks/s][A
Processed prompts:  20%|█▉        | 397/2000 [00:59<01:57, 13.61it/s, est. speed input: 6673.85 toks/s, output: 213.67 toks/s][A
Processed prompts:  21%|██        | 415/2000 [01:01<01:58, 13.41it/s, est. speed input: 6835.62 toks/s, output: 218.24 toks/s][A
Processed prompts:  22%|██▏       | 438/2000 [01:02<01:48, 14.34it/s, est. speed input: 7075.72 toks/s, output: 225.07 toks/s][A
Processed prompts:  23%|██▎       | 464/2000 [01:03<01:38, 15.67it/s, est. speed input: 7355.60 toks/s, output: 232.95 toks/s][A
Processed prompts:  24%|██▍       | 481/2000 [01:05<01:44, 14.59it/s, est. speed input: 7474.82 toks/s, output: 236.02 toks/s][A
Processed prompts:  25%|██▍       | 498/2000 [01:06<01:47, 13.96it/s, est. speed input: 7593.35 toks/s, output: 239.09 toks/s][A
Processed prompts:  25%|██▌       | 501/2000 [01:09<03:08,  7.97it/s, est. speed input: 7331.45 toks/s, output: 230.72 toks/s][A
Processed prompts:  25%|██▌       | 503/2000 [01:10<03:55,  6.35it/s, est. speed input: 7216.57 toks/s, output: 227.04 toks/s][A
Processed prompts:  25%|██▌       | 507/2000 [01:11<03:52,  6.42it/s, est. speed input: 7217.66 toks/s, output: 226.99 toks/s][A
Processed prompts:  26%|██▌       | 517/2000 [01:12<03:20,  7.39it/s, est. speed input: 7271.64 toks/s, output: 228.33 toks/s][A
Processed prompts:  26%|██▋       | 528/2000 [01:13<02:57,  8.29it/s, est. speed input: 7329.85 toks/s, output: 229.88 toks/s][A
Processed prompts:  27%|██▋       | 539/2000 [01:14<02:42,  9.00it/s, est. speed input: 7388.22 toks/s, output: 231.47 toks/s][A
Processed prompts:  28%|██▊       | 563/2000 [01:15<02:02, 11.70it/s, est. speed input: 7584.07 toks/s, output: 236.99 toks/s][A
Processed prompts:  29%|██▉       | 587/2000 [01:17<01:44, 13.53it/s, est. speed input: 7778.34 toks/s, output: 242.42 toks/s][A
Processed prompts:  30%|███       | 605/2000 [01:18<01:44, 13.33it/s, est. speed input: 7884.29 toks/s, output: 245.20 toks/s][A
Processed prompts:  31%|███       | 618/2000 [01:20<01:55, 11.93it/s, est. speed input: 7914.57 toks/s, output: 245.82 toks/s][A
Processed prompts:  32%|███▏      | 634/2000 [01:21<01:56, 11.77it/s, est. speed input: 7987.60 toks/s, output: 247.68 toks/s][A
Processed prompts:  33%|███▎      | 651/2000 [01:22<01:53, 11.90it/s, est. speed input: 8071.37 toks/s, output: 249.88 toks/s][A
Processed prompts:  34%|███▎      | 670/2000 [01:24<01:47, 12.42it/s, est. speed input: 8177.78 toks/s, output: 252.71 toks/s][A
Processed prompts:  35%|███▍      | 693/2000 [01:25<01:36, 13.49it/s, est. speed input: 8324.89 toks/s, output: 256.64 toks/s][A
Processed prompts:  36%|███▌      | 711/2000 [01:27<01:36, 13.34it/s, est. speed input: 8412.49 toks/s, output: 258.83 toks/s][A
Processed prompts:  36%|███▋      | 729/2000 [01:28<01:35, 13.24it/s, est. speed input: 8497.71 toks/s, output: 260.99 toks/s][A
Processed prompts:  37%|███▋      | 735/2000 [01:29<02:00, 10.47it/s, est. speed input: 8433.28 toks/s, output: 258.89 toks/s][A
Processed prompts:  37%|███▋      | 747/2000 [01:31<02:05,  9.95it/s, est. speed input: 8454.34 toks/s, output: 259.06 toks/s][A
Processed prompts:  38%|███▊      | 765/2000 [01:32<01:53, 10.87it/s, est. speed input: 8549.97 toks/s, output: 261.25 toks/s][A
Processed prompts:  39%|███▉      | 782/2000 [01:33<01:47, 11.30it/s, est. speed input: 8632.79 toks/s, output: 263.07 toks/s][A
Processed prompts:  40%|████      | 808/2000 [01:35<01:28, 13.41it/s, est. speed input: 8807.09 toks/s, output: 267.49 toks/s][A
Processed prompts:  41%|████▏     | 829/2000 [01:36<01:23, 13.95it/s, est. speed input: 8915.25 toks/s, output: 270.31 toks/s][A
Processed prompts:  42%|████▏     | 848/2000 [01:38<01:23, 13.86it/s, est. speed input: 8994.96 toks/s, output: 272.36 toks/s][A
Processed prompts:  43%|████▎     | 852/2000 [01:39<01:48, 10.57it/s, est. speed input: 8911.81 toks/s, output: 269.78 toks/s][A
Processed prompts:  43%|████▎     | 867/2000 [01:40<01:46, 10.61it/s, est. speed input: 8946.35 toks/s, output: 270.66 toks/s][A
Processed prompts:  44%|████▍     | 878/2000 [01:42<01:53,  9.85it/s, est. speed input: 8940.86 toks/s, output: 270.38 toks/s][A
Processed prompts:  45%|████▌     | 900/2000 [01:43<01:33, 11.71it/s, est. speed input: 9049.13 toks/s, output: 273.46 toks/s][A
Processed prompts:  46%|████▋     | 926/2000 [01:45<01:18, 13.69it/s, est. speed input: 9149.86 toks/s, output: 277.29 toks/s][A
Processed prompts:  47%|████▋     | 949/2000 [01:46<01:11, 14.61it/s, est. speed input: 9213.60 toks/s, output: 280.33 toks/s][A
Processed prompts:  48%|████▊     | 967/2000 [01:47<01:12, 14.16it/s, est. speed input: 9220.39 toks/s, output: 281.79 toks/s][A
Processed prompts:  49%|████▉     | 976/2000 [01:49<01:26, 11.86it/s, est. speed input: 9164.93 toks/s, output: 280.71 toks/s][A
Processed prompts:  49%|████▉     | 979/2000 [01:50<01:54,  8.89it/s, est. speed input: 9072.07 toks/s, output: 277.95 toks/s][A
Processed prompts:  50%|████▉     | 990/2000 [01:52<01:57,  8.63it/s, est. speed input: 9050.54 toks/s, output: 277.64 toks/s][A
Processed prompts:  50%|█████     | 1008/2000 [01:53<01:39,  9.93it/s, est. speed input: 9094.79 toks/s, output: 279.25 toks/s][A
Processed prompts:  51%|█████     | 1015/2000 [01:54<01:42,  9.59it/s, est. speed input: 9087.64 toks/s, output: 279.08 toks/s][A
Processed prompts:  51%|█████     | 1021/2000 [01:54<01:43,  9.46it/s, est. speed input: 9085.10 toks/s, output: 279.11 toks/s][A
Processed prompts:  52%|█████▏    | 1031/2000 [01:56<01:43,  9.37it/s, est. speed input: 9084.02 toks/s, output: 279.21 toks/s][A
Processed prompts:  53%|█████▎    | 1051/2000 [01:57<01:24, 11.19it/s, est. speed input: 9152.92 toks/s, output: 281.28 toks/s][A
Processed prompts:  53%|█████▎    | 1068/2000 [01:58<01:20, 11.56it/s, est. speed input: 9197.25 toks/s, output: 282.46 toks/s][A
Processed prompts:  54%|█████▍    | 1086/2000 [02:00<01:16, 11.94it/s, est. speed input: 9248.39 toks/s, output: 283.80 toks/s][A
Processed prompts:  55%|█████▌    | 1103/2000 [02:01<01:14, 12.05it/s, est. speed input: 9291.96 toks/s, output: 284.93 toks/s][A
Processed prompts:  56%|█████▌    | 1121/2000 [02:03<01:11, 12.37it/s, est. speed input: 9343.49 toks/s, output: 286.27 toks/s][A
Processed prompts:  57%|█████▋    | 1138/2000 [02:04<01:09, 12.40it/s, est. speed input: 9386.47 toks/s, output: 287.37 toks/s][A
Processed prompts:  58%|█████▊    | 1154/2000 [02:05<01:10, 12.00it/s, est. speed input: 9414.70 toks/s, output: 288.03 toks/s][A
Processed prompts:  58%|█████▊    | 1162/2000 [02:07<01:19, 10.49it/s, est. speed input: 9391.09 toks/s, output: 287.22 toks/s][A
Processed prompts:  59%|█████▉    | 1175/2000 [02:08<01:16, 10.82it/s, est. speed input: 9418.15 toks/s, output: 287.94 toks/s][A
Processed prompts:  59%|█████▉    | 1188/2000 [02:09<01:13, 11.04it/s, est. speed input: 9443.46 toks/s, output: 288.66 toks/s][A
Processed prompts:  60%|██████    | 1206/2000 [02:10<01:08, 11.63it/s, est. speed input: 9488.67 toks/s, output: 289.95 toks/s][A
Processed prompts:  61%|██████    | 1223/2000 [02:12<01:05, 11.92it/s, est. speed input: 9521.07 toks/s, output: 291.09 toks/s][A
Processed prompts:  62%|██████▏   | 1241/2000 [02:13<01:01, 12.35it/s, est. speed input: 9525.04 toks/s, output: 292.42 toks/s][A
Processed prompts:  63%|██████▎   | 1254/2000 [02:14<01:03, 11.67it/s, est. speed input: 9505.29 toks/s, output: 292.70 toks/s][A
Processed prompts:  64%|██████▎   | 1272/2000 [02:16<01:00, 12.07it/s, est. speed input: 9506.38 toks/s, output: 293.93 toks/s][A
Processed prompts:  64%|██████▍   | 1283/2000 [02:17<01:02, 11.54it/s, est. speed input: 9504.24 toks/s, output: 294.16 toks/s][A
Processed prompts:  65%|██████▌   | 1304/2000 [02:18<00:54, 12.80it/s, est. speed input: 9566.00 toks/s, output: 296.17 toks/s][A
Processed prompts:  66%|██████▌   | 1321/2000 [02:19<00:53, 12.73it/s, est. speed input: 9603.04 toks/s, output: 297.20 toks/s][A
Processed prompts:  67%|██████▋   | 1338/2000 [02:21<00:52, 12.54it/s, est. speed input: 9632.81 toks/s, output: 298.09 toks/s][A
Processed prompts:  68%|██████▊   | 1354/2000 [02:22<00:52, 12.34it/s, est. speed input: 9622.44 toks/s, output: 298.87 toks/s][A
Processed prompts:  69%|██████▊   | 1373/2000 [02:23<00:48, 12.87it/s, est. speed input: 9616.38 toks/s, output: 300.29 toks/s][A
Processed prompts:  70%|██████▉   | 1390/2000 [02:25<00:48, 12.69it/s, est. speed input: 9599.38 toks/s, output: 301.16 toks/s][A
Processed prompts:  70%|███████   | 1405/2000 [02:26<00:48, 12.24it/s, est. speed input: 9591.55 toks/s, output: 301.68 toks/s][A
Processed prompts:  71%|███████   | 1422/2000 [02:28<00:46, 12.34it/s, est. speed input: 9625.97 toks/s, output: 302.60 toks/s][A
Processed prompts:  72%|███████▏  | 1439/2000 [02:29<00:45, 12.44it/s, est. speed input: 9660.79 toks/s, output: 303.53 toks/s][A
Processed prompts:  73%|███████▎  | 1467/2000 [02:30<00:36, 14.75it/s, est. speed input: 9745.45 toks/s, output: 306.57 toks/s][A
Processed prompts:  74%|███████▍  | 1489/2000 [02:32<00:33, 15.20it/s, est. speed input: 9774.34 toks/s, output: 308.34 toks/s][A
Processed prompts:  75%|███████▌  | 1506/2000 [02:33<00:34, 14.41it/s, est. speed input: 9770.51 toks/s, output: 309.06 toks/s][A
Processed prompts:  76%|███████▌  | 1515/2000 [02:34<00:40, 12.05it/s, est. speed input: 9727.41 toks/s, output: 308.13 toks/s][A
Processed prompts:  76%|███████▌  | 1519/2000 [02:35<00:43, 10.94it/s, est. speed input: 9703.62 toks/s, output: 307.57 toks/s][A
Processed prompts:  76%|███████▌  | 1521/2000 [02:35<00:44, 10.79it/s, est. speed input: 9701.30 toks/s, output: 307.55 toks/s][A
Processed prompts:  77%|███████▋  | 1531/2000 [02:35<00:31, 14.85it/s, est. speed input: 9762.91 toks/s, output: 309.44 toks/s][A
Processed prompts:  78%|███████▊  | 1559/2000 [02:35<00:14, 31.20it/s, est. speed input: 9924.68 toks/s, output: 314.95 toks/s][A
Processed prompts:  80%|███████▉  | 1592/2000 [02:36<00:07, 55.36it/s, est. speed input: 10108.46 toks/s, output: 321.42 toks/s][A
Processed prompts:  81%|████████▏ | 1627/2000 [02:36<00:04, 79.74it/s, est. speed input: 10296.37 toks/s, output: 328.18 toks/s][A
Processed prompts:  82%|████████▏ | 1645/2000 [02:36<00:04, 84.34it/s, est. speed input: 10383.57 toks/s, output: 331.49 toks/s][A
Processed prompts:  83%|████████▎ | 1668/2000 [02:36<00:03, 94.46it/s, est. speed input: 10540.92 toks/s, output: 335.89 toks/s][A
Processed prompts:  85%|████████▍ | 1692/2000 [02:36<00:02, 106.41it/s, est. speed input: 10698.64 toks/s, output: 340.57 toks/s][A
Processed prompts:  87%|████████▋ | 1738/2000 [02:36<00:01, 151.88it/s, est. speed input: 10934.81 toks/s, output: 349.70 toks/s][A
Processed prompts:  89%|████████▉ | 1783/2000 [02:37<00:01, 195.78it/s, est. speed input: 11139.30 toks/s, output: 358.44 toks/s][A
Processed prompts:  91%|█████████▏| 1827/2000 [02:37<00:00, 240.13it/s, est. speed input: 11344.71 toks/s, output: 366.80 toks/s][A
Processed prompts:  94%|█████████▍| 1879/2000 [02:37<00:00, 280.69it/s, est. speed input: 11597.68 toks/s, output: 376.48 toks/s][A
Processed prompts:  96%|█████████▋| 1929/2000 [02:37<00:00, 323.53it/s, est. speed input: 11808.14 toks/s, output: 385.84 toks/s][A
Processed prompts:  99%|█████████▉| 1983/2000 [02:37<00:00, 356.23it/s, est. speed input: 12136.83 toks/s, output: 396.18 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:37<00:00, 12.70it/s, est. speed input: 12250.10 toks/s, output: 399.48 toks/s] 
2025-07-03 00:11:43,442 - INFO - 
Progress: 12000 items processed
2025-07-03 00:11:43,443 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:11:43,443 - INFO -   accuracy: 0.970
2025-07-03 00:11:43,443 - INFO -   macro_f1: 0.492
2025-07-03 00:11:43,443 - INFO -   weighted_f1: 0.985
2025-07-03 00:11:43,443 - INFO - 
OVERALL metrics:
2025-07-03 00:11:43,443 - INFO -   exact_match_ratio: 0.970
2025-07-03 00:11:43,443 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:11:43,443 - INFO -   accuracy: 0.970
2025-07-03 00:11:43,443 - INFO -   macro_f1: 0.197
2025-07-03 00:11:43,443 - INFO -   weighted_f1: 0.985
2025-07-03 00:11:43,443 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:11:43,443 - INFO -   accuracy: 0.970
2025-07-03 00:11:43,443 - INFO -   macro_f1: 0.246
2025-07-03 00:11:43,443 - INFO -   weighted_f1: 0.985
 50%|█████     | 6/12 [16:58<16:51, 168.52s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:28<15:57:19, 28.73s/it, est. speed input: 37.03 toks/s, output: 1.04 toks/s][A
Processed prompts:   0%|          | 3/2000 [00:29<4:12:46,  7.59s/it, est. speed input: 109.59 toks/s, output: 3.09 toks/s][A
Processed prompts:   0%|          | 6/2000 [00:29<1:40:42,  3.03s/it, est. speed input: 216.08 toks/s, output: 6.18 toks/s][A
Processed prompts:   1%|          | 12/2000 [00:30<38:46,  1.17s/it, est. speed input: 422.50 toks/s, output: 12.17 toks/s][A
Processed prompts:   2%|▏         | 32/2000 [00:31<11:11,  2.93it/s, est. speed input: 1077.36 toks/s, output: 32.34 toks/s][A
Processed prompts:   3%|▎         | 58/2000 [00:32<05:36,  5.77it/s, est. speed input: 1873.35 toks/s, output: 56.15 toks/s][A
Processed prompts:   4%|▍         | 84/2000 [00:34<03:46,  8.48it/s, est. speed input: 2604.93 toks/s, output: 77.91 toks/s][A
Processed prompts:   5%|▌         | 107/2000 [00:35<03:01, 10.41it/s, est. speed input: 3193.93 toks/s, output: 95.28 toks/s][A
Processed prompts:   6%|▋         | 125/2000 [00:36<02:48, 11.15it/s, est. speed input: 3537.26 toks/s, output: 106.93 toks/s][A
Processed prompts:   7%|▋         | 143/2000 [00:38<02:38, 11.68it/s, est. speed input: 3704.00 toks/s, output: 117.68 toks/s][A
Processed prompts:   8%|▊         | 161/2000 [00:39<02:30, 12.18it/s, est. speed input: 3863.72 toks/s, output: 127.82 toks/s][A
Processed prompts:   9%|▉         | 178/2000 [00:41<02:27, 12.36it/s, est. speed input: 3997.91 toks/s, output: 136.56 toks/s][A
Processed prompts:  10%|▉         | 195/2000 [00:42<02:25, 12.38it/s, est. speed input: 4357.40 toks/s, output: 144.60 toks/s][A
Processed prompts:  11%|█         | 213/2000 [00:43<02:20, 12.73it/s, est. speed input: 4759.08 toks/s, output: 152.99 toks/s][A
Processed prompts:  12%|█▏        | 230/2000 [00:45<02:18, 12.77it/s, est. speed input: 5108.82 toks/s, output: 160.20 toks/s][A
Processed prompts:  12%|█▏        | 241/2000 [00:47<03:20,  8.79it/s, est. speed input: 5104.56 toks/s, output: 158.33 toks/s][A
Processed prompts:  13%|█▎        | 251/2000 [00:49<03:26,  8.46it/s, est. speed input: 5093.08 toks/s, output: 160.67 toks/s][A
Processed prompts:  13%|█▎        | 269/2000 [00:50<02:57,  9.78it/s, est. speed input: 5181.02 toks/s, output: 168.14 toks/s][A
Processed prompts:  15%|█▌        | 300/2000 [00:51<02:08, 13.28it/s, est. speed input: 5448.01 toks/s, output: 182.78 toks/s][A
Processed prompts:  17%|█▋        | 333/2000 [00:53<01:41, 16.46it/s, est. speed input: 5731.09 toks/s, output: 198.11 toks/s][A
Processed prompts:  17%|█▋        | 349/2000 [00:54<01:48, 15.21it/s, est. speed input: 5804.31 toks/s, output: 202.42 toks/s][A
Processed prompts:  18%|█▊        | 366/2000 [00:55<01:52, 14.50it/s, est. speed input: 5931.53 toks/s, output: 207.07 toks/s][A
Processed prompts:  19%|█▉        | 384/2000 [00:57<01:54, 14.11it/s, est. speed input: 6109.52 toks/s, output: 211.92 toks/s][A
Processed prompts:  20%|██        | 404/2000 [00:58<01:50, 14.42it/s, est. speed input: 6320.78 toks/s, output: 217.79 toks/s][A
Processed prompts:  21%|██        | 421/2000 [00:59<01:53, 13.97it/s, est. speed input: 6469.15 toks/s, output: 221.80 toks/s][A
Processed prompts:  22%|██▏       | 438/2000 [01:01<01:56, 13.42it/s, est. speed input: 6498.72 toks/s, output: 225.36 toks/s][A
Processed prompts:  23%|██▎       | 455/2000 [01:02<01:56, 13.21it/s, est. speed input: 6532.96 toks/s, output: 228.98 toks/s][A
Processed prompts:  24%|██▍       | 479/2000 [01:03<01:44, 14.60it/s, est. speed input: 6641.41 toks/s, output: 235.73 toks/s][A
Processed prompts:  25%|██▍       | 494/2000 [01:05<01:50, 13.58it/s, est. speed input: 6656.36 toks/s, output: 237.93 toks/s][A
Processed prompts:  25%|██▍       | 499/2000 [01:06<02:22, 10.52it/s, est. speed input: 6567.34 toks/s, output: 235.32 toks/s][A
Processed prompts:  25%|██▌       | 502/2000 [01:07<03:05,  8.07it/s, est. speed input: 6468.14 toks/s, output: 232.05 toks/s][A
Processed prompts:  25%|██▌       | 505/2000 [01:09<03:55,  6.34it/s, est. speed input: 6372.59 toks/s, output: 228.90 toks/s][A
Processed prompts:  26%|██▌       | 510/2000 [01:10<03:53,  6.38it/s, est. speed input: 6351.60 toks/s, output: 228.55 toks/s][A
Processed prompts:  26%|██▋       | 525/2000 [01:11<03:04,  8.01it/s, est. speed input: 6378.01 toks/s, output: 230.67 toks/s][A
Processed prompts:  27%|██▋       | 532/2000 [01:12<02:54,  8.40it/s, est. speed input: 6381.75 toks/s, output: 231.36 toks/s][A
Processed prompts:  27%|██▋       | 544/2000 [01:13<02:36,  9.33it/s, est. speed input: 6403.00 toks/s, output: 233.17 toks/s][A
Processed prompts:  28%|██▊       | 563/2000 [01:14<02:10, 11.02it/s, est. speed input: 6460.20 toks/s, output: 236.84 toks/s][A
Processed prompts:  29%|██▉       | 582/2000 [01:15<01:57, 12.07it/s, est. speed input: 6516.38 toks/s, output: 240.40 toks/s][A
Processed prompts:  30%|██▉       | 595/2000 [01:17<02:05, 11.15it/s, est. speed input: 6513.39 toks/s, output: 241.31 toks/s][A
Processed prompts:  30%|███       | 600/2000 [01:17<02:13, 10.50it/s, est. speed input: 6501.92 toks/s, output: 241.28 toks/s][A
Processed prompts:  31%|███       | 613/2000 [01:18<02:07, 10.86it/s, est. speed input: 6515.55 toks/s, output: 243.12 toks/s][A
Processed prompts:  32%|███▏      | 631/2000 [01:20<01:57, 11.68it/s, est. speed input: 6547.64 toks/s, output: 246.16 toks/s][A
Processed prompts:  32%|███▏      | 649/2000 [01:21<01:52, 12.03it/s, est. speed input: 6574.98 toks/s, output: 248.92 toks/s][A
Processed prompts:  33%|███▎      | 664/2000 [01:23<01:54, 11.67it/s, est. speed input: 6586.15 toks/s, output: 250.57 toks/s][A
Processed prompts:  34%|███▍      | 681/2000 [01:24<01:50, 11.88it/s, est. speed input: 6628.62 toks/s, output: 252.93 toks/s][A
Processed prompts:  35%|███▍      | 698/2000 [01:25<01:48, 12.02it/s, est. speed input: 6669.22 toks/s, output: 255.20 toks/s][A
Processed prompts:  36%|███▌      | 720/2000 [01:27<01:37, 13.06it/s, est. speed input: 6748.55 toks/s, output: 259.02 toks/s][A
Processed prompts:  37%|███▋      | 735/2000 [01:28<01:42, 12.40it/s, est. speed input: 6771.86 toks/s, output: 260.36 toks/s][A
Processed prompts:  38%|███▊      | 752/2000 [01:30<01:40, 12.36it/s, est. speed input: 6811.67 toks/s, output: 262.36 toks/s][A
Processed prompts:  39%|███▉      | 781/2000 [01:31<01:22, 14.85it/s, est. speed input: 6978.12 toks/s, output: 268.17 toks/s][A
Processed prompts:  40%|████      | 809/2000 [01:32<01:11, 16.56it/s, est. speed input: 7153.19 toks/s, output: 273.59 toks/s][A
Processed prompts:  41%|████▏     | 825/2000 [01:34<01:17, 15.12it/s, est. speed input: 7220.72 toks/s, output: 274.80 toks/s][A
Processed prompts:  42%|████▏     | 835/2000 [01:35<01:31, 12.78it/s, est. speed input: 7222.51 toks/s, output: 274.10 toks/s][A
Processed prompts:  42%|████▏     | 849/2000 [01:36<01:36, 11.88it/s, est. speed input: 7237.97 toks/s, output: 274.58 toks/s][A
Processed prompts:  43%|████▎     | 868/2000 [01:38<01:30, 12.46it/s, est. speed input: 7280.22 toks/s, output: 276.74 toks/s][A
Processed prompts:  44%|████▍     | 885/2000 [01:39<01:29, 12.43it/s, est. speed input: 7306.52 toks/s, output: 278.20 toks/s][A
Processed prompts:  45%|████▌     | 901/2000 [01:41<01:30, 12.18it/s, est. speed input: 7329.81 toks/s, output: 279.31 toks/s][A
Processed prompts:  46%|████▌     | 911/2000 [01:42<01:42, 10.65it/s, est. speed input: 7352.77 toks/s, output: 278.53 toks/s][A
Processed prompts:  46%|████▋     | 925/2000 [01:43<01:42, 10.53it/s, est. speed input: 7427.30 toks/s, output: 279.10 toks/s][A
Processed prompts:  47%|████▋     | 943/2000 [01:45<01:33, 11.32it/s, est. speed input: 7547.56 toks/s, output: 280.87 toks/s][A
Processed prompts:  48%|████▊     | 953/2000 [01:46<01:44, 10.06it/s, est. speed input: 7567.76 toks/s, output: 280.17 toks/s][A
Processed prompts:  48%|████▊     | 967/2000 [01:47<01:41, 10.13it/s, est. speed input: 7602.89 toks/s, output: 280.81 toks/s][A
Processed prompts:  49%|████▉     | 982/2000 [01:49<01:37, 10.39it/s, est. speed input: 7609.86 toks/s, output: 281.70 toks/s][A
Processed prompts:  50%|█████     | 1002/2000 [01:50<01:25, 11.66it/s, est. speed input: 7650.63 toks/s, output: 284.03 toks/s][A
Processed prompts:  51%|█████     | 1011/2000 [01:51<01:34, 10.44it/s, est. speed input: 7625.97 toks/s, output: 283.45 toks/s][A
Processed prompts:  51%|█████     | 1023/2000 [01:52<01:31, 10.67it/s, est. speed input: 7633.82 toks/s, output: 284.19 toks/s][A
Processed prompts:  52%|█████▏    | 1036/2000 [01:54<01:28, 10.95it/s, est. speed input: 7644.40 toks/s, output: 285.05 toks/s][A
Processed prompts:  53%|█████▎    | 1056/2000 [01:55<01:17, 12.13it/s, est. speed input: 7681.76 toks/s, output: 287.10 toks/s][A
Processed prompts:  53%|█████▎    | 1069/2000 [01:56<01:23, 11.09it/s, est. speed input: 7669.67 toks/s, output: 287.03 toks/s][A
Processed prompts:  54%|█████▍    | 1075/2000 [01:57<01:27, 10.58it/s, est. speed input: 7660.73 toks/s, output: 286.91 toks/s][A
Processed prompts:  55%|█████▍    | 1092/2000 [01:59<01:30,  9.99it/s, est. speed input: 7647.82 toks/s, output: 287.03 toks/s][A
Processed prompts:  55%|█████▌    | 1100/2000 [02:00<01:28, 10.13it/s, est. speed input: 7649.65 toks/s, output: 287.38 toks/s][A
Processed prompts:  56%|█████▌    | 1113/2000 [02:01<01:23, 10.57it/s, est. speed input: 7658.99 toks/s, output: 288.17 toks/s][A
Processed prompts:  56%|█████▋    | 1129/2000 [02:02<01:21, 10.69it/s, est. speed input: 7664.49 toks/s, output: 288.89 toks/s][A
Processed prompts:  57%|█████▋    | 1146/2000 [02:04<01:16, 11.20it/s, est. speed input: 7696.36 toks/s, output: 290.04 toks/s][A
Processed prompts:  58%|█████▊    | 1165/2000 [02:05<01:09, 11.95it/s, est. speed input: 7771.41 toks/s, output: 291.62 toks/s][A
Processed prompts:  59%|█████▉    | 1180/2000 [02:07<01:10, 11.58it/s, est. speed input: 7811.65 toks/s, output: 292.19 toks/s][A
Processed prompts:  60%|██████    | 1204/2000 [02:08<01:00, 13.25it/s, est. speed input: 7925.10 toks/s, output: 294.89 toks/s][A
Processed prompts:  61%|██████▏   | 1227/2000 [02:09<00:53, 14.33it/s, est. speed input: 8029.71 toks/s, output: 297.33 toks/s][A
Processed prompts:  62%|██████▏   | 1245/2000 [02:11<00:54, 13.96it/s, est. speed input: 8091.73 toks/s, output: 298.47 toks/s][A
Processed prompts:  63%|██████▎   | 1260/2000 [02:12<00:57, 12.97it/s, est. speed input: 8126.71 toks/s, output: 298.84 toks/s][A
Processed prompts:  63%|██████▎   | 1262/2000 [02:13<01:15,  9.77it/s, est. speed input: 8062.21 toks/s, output: 296.50 toks/s][A
Processed prompts:  64%|██████▍   | 1280/2000 [02:15<01:06, 10.81it/s, est. speed input: 8085.04 toks/s, output: 297.76 toks/s][A
Processed prompts:  65%|██████▍   | 1296/2000 [02:16<01:03, 11.09it/s, est. speed input: 8096.02 toks/s, output: 298.54 toks/s][A
Processed prompts:  66%|██████▌   | 1321/2000 [02:17<00:51, 13.18it/s, est. speed input: 8173.17 toks/s, output: 301.26 toks/s][A
Processed prompts:  67%|██████▋   | 1346/2000 [02:19<00:44, 14.76it/s, est. speed input: 8270.27 toks/s, output: 303.95 toks/s][A
Processed prompts:  68%|██████▊   | 1363/2000 [02:20<00:45, 14.09it/s, est. speed input: 8318.96 toks/s, output: 304.76 toks/s][A
Processed prompts:  69%|██████▉   | 1380/2000 [02:22<00:45, 13.52it/s, est. speed input: 8364.86 toks/s, output: 305.48 toks/s][A
Processed prompts:  70%|██████▉   | 1396/2000 [02:23<00:46, 13.02it/s, est. speed input: 8369.06 toks/s, output: 306.06 toks/s][A
Processed prompts:  71%|███████   | 1413/2000 [02:24<00:45, 12.85it/s, est. speed input: 8377.48 toks/s, output: 306.82 toks/s][A
Processed prompts:  72%|███████▏  | 1430/2000 [02:26<00:44, 12.76it/s, est. speed input: 8386.32 toks/s, output: 307.58 toks/s][A
Processed prompts:  73%|███████▎  | 1451/2000 [02:27<00:40, 13.47it/s, est. speed input: 8404.34 toks/s, output: 309.06 toks/s][A
Processed prompts:  73%|███████▎  | 1465/2000 [02:28<00:42, 12.55it/s, est. speed input: 8386.74 toks/s, output: 309.15 toks/s][A
Processed prompts:  74%|███████▍  | 1483/2000 [02:30<00:40, 12.74it/s, est. speed input: 8385.34 toks/s, output: 310.05 toks/s][A
Processed prompts:  75%|███████▍  | 1496/2000 [02:31<00:43, 11.62it/s, est. speed input: 8360.05 toks/s, output: 309.80 toks/s][A
Processed prompts:  75%|███████▌  | 1501/2000 [02:33<00:54,  9.23it/s, est. speed input: 8308.15 toks/s, output: 308.04 toks/s][A
Processed prompts:  76%|███████▌  | 1511/2000 [02:33<00:45, 10.76it/s, est. speed input: 8354.25 toks/s, output: 309.18 toks/s][A
Processed prompts:  76%|███████▌  | 1523/2000 [02:33<00:32, 14.73it/s, est. speed input: 8432.77 toks/s, output: 311.44 toks/s][A
Processed prompts:  77%|███████▋  | 1535/2000 [02:33<00:23, 19.87it/s, est. speed input: 8511.86 toks/s, output: 313.71 toks/s][A
Processed prompts:  78%|███████▊  | 1551/2000 [02:33<00:15, 28.99it/s, est. speed input: 8619.58 toks/s, output: 316.82 toks/s][A
Processed prompts:  78%|███████▊  | 1560/2000 [02:33<00:12, 34.07it/s, est. speed input: 8677.51 toks/s, output: 318.48 toks/s][A
Processed prompts:  78%|███████▊  | 1569/2000 [02:34<00:10, 40.02it/s, est. speed input: 8739.32 toks/s, output: 320.18 toks/s][A
Processed prompts:  79%|███████▉  | 1579/2000 [02:34<00:08, 47.84it/s, est. speed input: 8807.85 toks/s, output: 322.08 toks/s][A
Processed prompts:  80%|███████▉  | 1594/2000 [02:34<00:06, 63.74it/s, est. speed input: 8913.64 toks/s, output: 325.07 toks/s][A
Processed prompts:  80%|████████  | 1608/2000 [02:34<00:05, 77.62it/s, est. speed input: 9018.88 toks/s, output: 327.74 toks/s][A
Processed prompts:  83%|████████▎ | 1659/2000 [02:34<00:02, 146.56it/s, est. speed input: 9418.69 toks/s, output: 337.88 toks/s][A
Processed prompts:  84%|████████▍ | 1681/2000 [02:34<00:02, 143.15it/s, est. speed input: 9589.97 toks/s, output: 341.98 toks/s][A
Processed prompts:  85%|████████▍ | 1698/2000 [02:34<00:02, 131.79it/s, est. speed input: 9661.34 toks/s, output: 345.23 toks/s][A
Processed prompts:  87%|████████▋ | 1745/2000 [02:35<00:01, 184.69it/s, est. speed input: 9867.04 toks/s, output: 354.70 toks/s][A
Processed prompts:  90%|████████▉ | 1792/2000 [02:35<00:00, 238.29it/s, est. speed input: 10055.93 toks/s, output: 364.03 toks/s][A
Processed prompts:  91%|█████████ | 1819/2000 [02:35<00:00, 240.92it/s, est. speed input: 10155.69 toks/s, output: 369.32 toks/s][A
Processed prompts:  94%|█████████▍| 1880/2000 [02:35<00:00, 302.82it/s, est. speed input: 10403.51 toks/s, output: 381.43 toks/s][A
Processed prompts:  96%|█████████▋| 1930/2000 [02:35<00:00, 344.66it/s, est. speed input: 10639.89 toks/s, output: 391.14 toks/s][A
Processed prompts:  99%|█████████▉| 1980/2000 [02:35<00:00, 368.91it/s, est. speed input: 10889.67 toks/s, output: 400.86 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:35<00:00, 12.85it/s, est. speed input: 11037.76 toks/s, output: 404.97 toks/s] 
2025-07-03 00:14:28,305 - INFO - 
Progress: 14000 items processed
2025-07-03 00:14:28,305 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:14:28,305 - INFO -   accuracy: 0.853
2025-07-03 00:14:28,305 - INFO -   macro_f1: 0.556
2025-07-03 00:14:28,305 - INFO -   weighted_f1: 0.819
2025-07-03 00:14:28,305 - INFO - 
OVERALL metrics:
2025-07-03 00:14:28,305 - INFO -   exact_match_ratio: 0.840
2025-07-03 00:14:28,305 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:14:28,305 - INFO -   accuracy: 0.849
2025-07-03 00:14:28,305 - INFO -   macro_f1: 0.270
2025-07-03 00:14:28,305 - INFO -   weighted_f1: 0.810
2025-07-03 00:14:28,305 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:14:28,305 - INFO -   accuracy: 0.840
2025-07-03 00:14:28,305 - INFO -   macro_f1: 0.202
2025-07-03 00:14:28,305 - INFO -   weighted_f1: 0.799
 58%|█████▊    | 7/12 [19:43<13:56, 167.32s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:27<15:32:40, 27.99s/it, est. speed input: 42.15 toks/s, output: 1.18 toks/s][A
Processed prompts:   0%|          | 9/2000 [00:28<1:17:43,  2.34s/it, est. speed input: 286.63 toks/s, output: 9.66 toks/s][A
Processed prompts:   2%|▏         | 34/2000 [00:30<16:31,  1.98it/s, est. speed input: 1041.16 toks/s, output: 34.94 toks/s][A
Processed prompts:   3%|▎         | 66/2000 [00:31<07:23,  4.36it/s, est. speed input: 2018.27 toks/s, output: 65.57 toks/s][A
Processed prompts:   5%|▍         | 91/2000 [00:32<05:04,  6.27it/s, est. speed input: 2649.90 toks/s, output: 86.84 toks/s][A
Processed prompts:   5%|▍         | 94/2000 [00:34<05:40,  5.59it/s, est. speed input: 2611.14 toks/s, output: 86.13 toks/s][A
Processed prompts:   5%|▌         | 101/2000 [00:35<05:49,  5.44it/s, est. speed input: 2654.25 toks/s, output: 88.90 toks/s][A
Processed prompts:   6%|▌         | 120/2000 [00:37<04:20,  7.21it/s, est. speed input: 2875.00 toks/s, output: 101.97 toks/s][A
Processed prompts:   7%|▋         | 149/2000 [00:38<02:57, 10.43it/s, est. speed input: 3377.47 toks/s, output: 121.86 toks/s][A
Processed prompts:   9%|▉         | 181/2000 [00:39<02:14, 13.57it/s, est. speed input: 3936.13 toks/s, output: 142.53 toks/s][A
Processed prompts:  10%|█         | 202/2000 [00:41<02:07, 14.11it/s, est. speed input: 4308.02 toks/s, output: 153.44 toks/s][A
Processed prompts:  11%|█         | 217/2000 [00:42<02:13, 13.32it/s, est. speed input: 4548.98 toks/s, output: 159.35 toks/s][A
Processed prompts:  11%|█         | 220/2000 [00:43<02:54, 10.22it/s, est. speed input: 4468.33 toks/s, output: 156.58 toks/s][A
Processed prompts:  12%|█▏        | 242/2000 [00:45<02:29, 11.79it/s, est. speed input: 4721.13 toks/s, output: 166.81 toks/s][A
Processed prompts:  13%|█▎        | 260/2000 [00:46<02:22, 12.20it/s, est. speed input: 4877.83 toks/s, output: 173.89 toks/s][A
Processed prompts:  14%|█▍        | 278/2000 [00:48<02:17, 12.50it/s, est. speed input: 5026.32 toks/s, output: 180.57 toks/s][A
Processed prompts:  14%|█▍        | 281/2000 [00:49<03:01,  9.46it/s, est. speed input: 4939.88 toks/s, output: 177.47 toks/s][A
Processed prompts:  15%|█▍        | 291/2000 [00:50<03:16,  8.68it/s, est. speed input: 5014.43 toks/s, output: 178.75 toks/s][A
Processed prompts:  15%|█▌        | 307/2000 [00:52<02:57,  9.55it/s, est. speed input: 5217.36 toks/s, output: 183.90 toks/s][A
Processed prompts:  16%|█▋        | 326/2000 [00:53<02:34, 10.82it/s, est. speed input: 5470.80 toks/s, output: 190.60 toks/s][A
Processed prompts:  17%|█▋        | 340/2000 [00:55<02:38, 10.47it/s, est. speed input: 5573.34 toks/s, output: 193.78 toks/s][A
Processed prompts:  18%|█▊        | 363/2000 [00:56<02:13, 12.31it/s, est. speed input: 5771.14 toks/s, output: 202.18 toks/s][A
Processed prompts:  19%|█▉        | 383/2000 [00:57<02:04, 12.98it/s, est. speed input: 5884.84 toks/s, output: 208.34 toks/s][A
Processed prompts:  21%|██        | 414/2000 [00:59<01:40, 15.77it/s, est. speed input: 6105.14 toks/s, output: 219.84 toks/s][A
Processed prompts:  22%|██▏       | 432/2000 [01:00<01:46, 14.78it/s, est. speed input: 6148.52 toks/s, output: 223.77 toks/s][A
Processed prompts:  22%|██▏       | 446/2000 [01:02<01:55, 13.42it/s, est. speed input: 6149.47 toks/s, output: 225.90 toks/s][A
Processed prompts:  23%|██▎       | 454/2000 [01:03<02:18, 11.13it/s, est. speed input: 6092.04 toks/s, output: 224.97 toks/s][A
Processed prompts:  23%|██▎       | 460/2000 [01:04<02:52,  8.94it/s, est. speed input: 6018.64 toks/s, output: 222.85 toks/s][A
Processed prompts:  24%|██▍       | 482/2000 [01:06<02:18, 10.99it/s, est. speed input: 6131.50 toks/s, output: 228.88 toks/s][A
Processed prompts:  25%|██▍       | 499/2000 [01:07<02:12, 11.37it/s, est. speed input: 6192.76 toks/s, output: 232.23 toks/s][A
Processed prompts:  25%|██▌       | 501/2000 [01:09<03:27,  7.23it/s, est. speed input: 6028.33 toks/s, output: 225.97 toks/s][A
Processed prompts:  25%|██▌       | 504/2000 [01:10<03:33,  7.02it/s, est. speed input: 6022.93 toks/s, output: 225.55 toks/s][A
Processed prompts:  26%|██▌       | 520/2000 [01:11<02:52,  8.58it/s, est. speed input: 6114.31 toks/s, output: 228.33 toks/s][A
Processed prompts:  27%|██▋       | 549/2000 [01:13<01:56, 12.47it/s, est. speed input: 6373.27 toks/s, output: 236.36 toks/s][A
Processed prompts:  28%|██▊       | 564/2000 [01:14<01:59, 12.01it/s, est. speed input: 6470.64 toks/s, output: 238.20 toks/s][A
Processed prompts:  29%|██▊       | 574/2000 [01:15<02:16, 10.47it/s, est. speed input: 6490.64 toks/s, output: 237.86 toks/s][A
Processed prompts:  29%|██▉       | 588/2000 [01:17<02:16, 10.36it/s, est. speed input: 6593.96 toks/s, output: 239.22 toks/s][A
Processed prompts:  30%|███       | 607/2000 [01:18<02:02, 11.39it/s, est. speed input: 6798.42 toks/s, output: 242.54 toks/s][A
Processed prompts:  31%|███▏      | 625/2000 [01:20<01:56, 11.77it/s, est. speed input: 6975.69 toks/s, output: 245.20 toks/s][A
Processed prompts:  32%|███▏      | 639/2000 [01:21<02:00, 11.29it/s, est. speed input: 7085.48 toks/s, output: 246.39 toks/s][A
Processed prompts:  32%|███▏      | 642/2000 [01:21<02:11, 10.35it/s, est. speed input: 7065.16 toks/s, output: 245.81 toks/s][A
Processed prompts:  33%|███▎      | 659/2000 [01:23<02:02, 10.98it/s, est. speed input: 7062.57 toks/s, output: 248.22 toks/s][A
Processed prompts:  34%|███▍      | 676/2000 [01:24<01:56, 11.39it/s, est. speed input: 7060.43 toks/s, output: 250.57 toks/s][A
Processed prompts:  35%|███▌      | 702/2000 [01:26<01:35, 13.56it/s, est. speed input: 7165.52 toks/s, output: 255.84 toks/s][A
Processed prompts:  36%|███▌      | 724/2000 [01:28<01:39, 12.86it/s, est. speed input: 7239.20 toks/s, output: 258.16 toks/s][A
Processed prompts:  37%|███▋      | 739/2000 [01:29<01:42, 12.30it/s, est. speed input: 7306.36 toks/s, output: 259.48 toks/s][A
Processed prompts:  37%|███▋      | 748/2000 [01:30<01:58, 10.54it/s, est. speed input: 7297.77 toks/s, output: 258.58 toks/s][A
Processed prompts:  38%|███▊      | 757/2000 [01:32<02:11,  9.45it/s, est. speed input: 7295.23 toks/s, output: 257.94 toks/s][A
Processed prompts:  39%|███▉      | 775/2000 [01:33<01:56, 10.52it/s, est. speed input: 7406.96 toks/s, output: 260.33 toks/s][A
Processed prompts:  40%|███▉      | 793/2000 [01:34<01:46, 11.28it/s, est. speed input: 7517.47 toks/s, output: 262.61 toks/s][A
Processed prompts:  41%|████      | 815/2000 [01:36<01:34, 12.58it/s, est. speed input: 7665.67 toks/s, output: 266.02 toks/s][A
Processed prompts:  42%|████▏     | 841/2000 [01:37<01:19, 14.51it/s, est. speed input: 7851.02 toks/s, output: 270.68 toks/s][A
Processed prompts:  43%|████▎     | 859/2000 [01:39<01:20, 14.12it/s, est. speed input: 7936.19 toks/s, output: 272.58 toks/s][A
Processed prompts:  44%|████▍     | 876/2000 [01:40<01:22, 13.63it/s, est. speed input: 8008.60 toks/s, output: 274.12 toks/s][A
Processed prompts:  45%|████▍     | 891/2000 [01:41<01:26, 12.77it/s, est. speed input: 8056.50 toks/s, output: 274.96 toks/s][A
Processed prompts:  45%|████▌     | 907/2000 [01:43<01:27, 12.48it/s, est. speed input: 8115.23 toks/s, output: 276.16 toks/s][A
Processed prompts:  46%|████▋     | 926/2000 [01:44<01:23, 12.92it/s, est. speed input: 8202.74 toks/s, output: 278.20 toks/s][A
Processed prompts:  47%|████▋     | 938/2000 [01:45<01:31, 11.61it/s, est. speed input: 8215.50 toks/s, output: 278.07 toks/s][A
Processed prompts:  47%|████▋     | 941/2000 [01:47<01:55,  9.17it/s, est. speed input: 8155.51 toks/s, output: 275.89 toks/s][A
Processed prompts:  48%|████▊     | 959/2000 [01:48<01:40, 10.40it/s, est. speed input: 8248.70 toks/s, output: 277.73 toks/s][A
Processed prompts:  49%|████▉     | 976/2000 [01:49<01:32, 11.04it/s, est. speed input: 8328.85 toks/s, output: 279.24 toks/s][A
Processed prompts:  50%|████▉     | 994/2000 [01:51<01:26, 11.60it/s, est. speed input: 8415.31 toks/s, output: 280.90 toks/s][A
Processed prompts:  50%|█████     | 1004/2000 [01:52<01:30, 10.98it/s, est. speed input: 8418.72 toks/s, output: 280.99 toks/s][A
Processed prompts:  51%|█████     | 1020/2000 [01:53<01:25, 11.40it/s, est. speed input: 8410.81 toks/s, output: 282.15 toks/s][A
Processed prompts:  52%|█████▏    | 1037/2000 [01:55<01:22, 11.72it/s, est. speed input: 8402.65 toks/s, output: 283.37 toks/s][A
Processed prompts:  53%|█████▎    | 1062/2000 [01:56<01:08, 13.65it/s, est. speed input: 8444.76 toks/s, output: 286.56 toks/s][A
Processed prompts:  54%|█████▍    | 1081/2000 [01:57<01:06, 13.76it/s, est. speed input: 8461.05 toks/s, output: 288.16 toks/s][A
Processed prompts:  55%|█████▍    | 1099/2000 [01:59<01:06, 13.60it/s, est. speed input: 8476.83 toks/s, output: 289.43 toks/s][A
Processed prompts:  56%|█████▌    | 1115/2000 [02:00<01:08, 12.94it/s, est. speed input: 8477.32 toks/s, output: 290.13 toks/s][A
Processed prompts:  56%|█████▌    | 1117/2000 [02:01<01:27, 10.12it/s, est. speed input: 8413.97 toks/s, output: 288.07 toks/s][A
Processed prompts:  56%|█████▌    | 1124/2000 [02:02<01:28,  9.90it/s, est. speed input: 8413.95 toks/s, output: 288.05 toks/s][A
Processed prompts:  56%|█████▋    | 1125/2000 [02:02<01:35,  9.14it/s, est. speed input: 8400.18 toks/s, output: 287.55 toks/s][A
Processed prompts:  57%|█████▋    | 1143/2000 [02:04<01:19, 10.79it/s, est. speed input: 8462.88 toks/s, output: 289.12 toks/s][A
Processed prompts:  58%|█████▊    | 1153/2000 [02:05<01:20, 10.56it/s, est. speed input: 8480.35 toks/s, output: 289.36 toks/s][A
Processed prompts:  59%|█████▉    | 1176/2000 [02:06<01:04, 12.76it/s, est. speed input: 8581.22 toks/s, output: 291.99 toks/s][A
Processed prompts:  60%|██████    | 1203/2000 [02:07<00:52, 15.24it/s, est. speed input: 8715.32 toks/s, output: 295.62 toks/s][A
Processed prompts:  61%|██████▏   | 1225/2000 [02:09<00:49, 15.60it/s, est. speed input: 8805.68 toks/s, output: 297.89 toks/s][A
Processed prompts:  62%|██████▏   | 1241/2000 [02:11<01:10, 10.83it/s, est. speed input: 8750.98 toks/s, output: 295.44 toks/s][A
Processed prompts:  63%|██████▎   | 1257/2000 [02:13<01:07, 11.08it/s, est. speed input: 8789.85 toks/s, output: 296.28 toks/s][A
Processed prompts:  64%|██████▎   | 1273/2000 [02:14<01:04, 11.26it/s, est. speed input: 8827.39 toks/s, output: 297.09 toks/s][A
Processed prompts:  64%|██████▍   | 1290/2000 [02:16<01:01, 11.48it/s, est. speed input: 8868.78 toks/s, output: 298.00 toks/s][A
Processed prompts:  65%|██████▌   | 1308/2000 [02:17<00:57, 11.98it/s, est. speed input: 8894.08 toks/s, output: 299.24 toks/s][A
Processed prompts:  66%|██████▋   | 1327/2000 [02:18<00:53, 12.53it/s, est. speed input: 8889.95 toks/s, output: 300.65 toks/s][A
Processed prompts:  67%|██████▋   | 1342/2000 [02:20<00:54, 12.08it/s, est. speed input: 8868.62 toks/s, output: 301.14 toks/s][A
Processed prompts:  68%|██████▊   | 1357/2000 [02:21<00:55, 11.67it/s, est. speed input: 8845.57 toks/s, output: 301.56 toks/s][A
Processed prompts:  69%|██████▉   | 1375/2000 [02:22<00:51, 12.13it/s, est. speed input: 8841.27 toks/s, output: 302.72 toks/s][A
Processed prompts:  70%|██████▉   | 1392/2000 [02:24<00:49, 12.23it/s, est. speed input: 8832.73 toks/s, output: 303.63 toks/s][A
Processed prompts:  70%|███████   | 1409/2000 [02:25<00:48, 12.17it/s, est. speed input: 8821.63 toks/s, output: 304.43 toks/s][A
Processed prompts:  71%|███████   | 1418/2000 [02:26<00:54, 10.69it/s, est. speed input: 8783.37 toks/s, output: 303.72 toks/s][A
Processed prompts:  71%|███████▏  | 1426/2000 [02:27<00:54, 10.62it/s, est. speed input: 8776.17 toks/s, output: 303.92 toks/s][A
Processed prompts:  72%|███████▏  | 1441/2000 [02:28<00:50, 11.04it/s, est. speed input: 8777.08 toks/s, output: 304.68 toks/s][A
Processed prompts:  73%|███████▎  | 1467/2000 [02:30<00:39, 13.65it/s, est. speed input: 8828.96 toks/s, output: 307.49 toks/s][A
Processed prompts:  75%|███████▍  | 1494/2000 [02:31<00:32, 15.37it/s, est. speed input: 8880.69 toks/s, output: 310.33 toks/s][A
Processed prompts:  76%|███████▌  | 1520/2000 [02:33<00:29, 16.52it/s, est. speed input: 8933.15 toks/s, output: 312.93 toks/s][A
Processed prompts:  77%|███████▋  | 1536/2000 [02:34<00:28, 16.52it/s, est. speed input: 8958.47 toks/s, output: 314.19 toks/s][A
Processed prompts:  77%|███████▋  | 1540/2000 [02:34<00:26, 17.27it/s, est. speed input: 8972.51 toks/s, output: 314.78 toks/s][A
Processed prompts:  78%|███████▊  | 1553/2000 [02:34<00:19, 22.39it/s, est. speed input: 9029.11 toks/s, output: 317.25 toks/s][A
Processed prompts:  78%|███████▊  | 1570/2000 [02:34<00:13, 31.40it/s, est. speed input: 9104.63 toks/s, output: 320.56 toks/s][A
Processed prompts:  80%|████████  | 1603/2000 [02:34<00:07, 51.51it/s, est. speed input: 9251.90 toks/s, output: 326.98 toks/s][A
Processed prompts:  81%|████████  | 1624/2000 [02:34<00:06, 61.81it/s, est. speed input: 9341.68 toks/s, output: 330.93 toks/s][A
Processed prompts:  82%|████████▏ | 1641/2000 [02:34<00:05, 68.04it/s, est. speed input: 9412.67 toks/s, output: 334.06 toks/s][A
Processed prompts:  83%|████████▎ | 1669/2000 [02:35<00:03, 87.33it/s, est. speed input: 9536.87 toks/s, output: 339.47 toks/s][A
Processed prompts:  85%|████████▌ | 1703/2000 [02:35<00:02, 113.98it/s, est. speed input: 9689.59 toks/s, output: 346.12 toks/s][A
Processed prompts:  87%|████████▋ | 1735/2000 [02:35<00:01, 135.31it/s, est. speed input: 9852.94 toks/s, output: 352.37 toks/s][A
Processed prompts:  88%|████████▊ | 1767/2000 [02:35<00:01, 159.93it/s, est. speed input: 10039.72 toks/s, output: 358.69 toks/s][A
Processed prompts:  90%|█████████ | 1802/2000 [02:35<00:01, 190.21it/s, est. speed input: 10261.72 toks/s, output: 365.66 toks/s][A
Processed prompts:  92%|█████████▏| 1849/2000 [02:35<00:00, 226.72it/s, est. speed input: 10554.60 toks/s, output: 374.98 toks/s][A
Processed prompts:  95%|█████████▍| 1893/2000 [02:35<00:00, 260.48it/s, est. speed input: 10758.28 toks/s, output: 383.80 toks/s][A
Processed prompts:  97%|█████████▋| 1934/2000 [02:36<00:00, 290.55it/s, est. speed input: 11014.10 toks/s, output: 392.10 toks/s][A
Processed prompts: 100%|█████████▉| 1990/2000 [02:36<00:00, 343.95it/s, est. speed input: 11415.34 toks/s, output: 403.59 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:36<00:00, 12.80it/s, est. speed input: 11492.71 toks/s, output: 405.70 toks/s] 
2025-07-03 00:17:14,227 - INFO - 
Progress: 16000 items processed
2025-07-03 00:17:14,227 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:17:14,227 - INFO -   accuracy: 0.760
2025-07-03 00:17:14,227 - INFO -   macro_f1: 0.525
2025-07-03 00:17:14,227 - INFO -   weighted_f1: 0.695
2025-07-03 00:17:14,227 - INFO - 
OVERALL metrics:
2025-07-03 00:17:14,227 - INFO -   exact_match_ratio: 0.740
2025-07-03 00:17:14,227 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:17:14,227 - INFO -   accuracy: 0.754
2025-07-03 00:17:14,227 - INFO -   macro_f1: 0.244
2025-07-03 00:17:14,227 - INFO -   weighted_f1: 0.683
2025-07-03 00:17:14,227 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:17:14,227 - INFO -   accuracy: 0.740
2025-07-03 00:17:14,227 - INFO -   macro_f1: 0.182
2025-07-03 00:17:14,227 - INFO -   weighted_f1: 0.661
 67%|██████▋   | 8/12 [22:29<11:07, 166.88s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:29<16:25:33, 29.58s/it, est. speed input: 40.87 toks/s, output: 1.12 toks/s][A
Processed prompts:   0%|          | 2/2000 [00:30<6:55:34, 12.48s/it, est. speed input: 80.43 toks/s, output: 2.19 toks/s] [A
Processed prompts:   1%|          | 21/2000 [00:31<26:32,  1.24it/s, est. speed input: 766.97 toks/s, output: 21.72 toks/s][A
Processed prompts:   2%|▏         | 44/2000 [00:32<11:12,  2.91it/s, est. speed input: 1507.83 toks/s, output: 43.40 toks/s][A
Processed prompts:   3%|▎         | 64/2000 [00:34<07:14,  4.45it/s, est. speed input: 2071.88 toks/s, output: 60.34 toks/s][A
Processed prompts:   5%|▍         | 99/2000 [00:35<04:05,  7.76it/s, est. speed input: 2851.01 toks/s, output: 88.76 toks/s][A
Processed prompts:   6%|▌         | 117/2000 [00:37<03:35,  8.74it/s, est. speed input: 3095.89 toks/s, output: 100.50 toks/s][A
Processed prompts:   7%|▋         | 135/2000 [00:38<03:13,  9.65it/s, est. speed input: 3324.92 toks/s, output: 111.42 toks/s][A
Processed prompts:   7%|▋         | 141/2000 [00:39<03:41,  8.38it/s, est. speed input: 3324.25 toks/s, output: 112.15 toks/s][A
Processed prompts:   8%|▊         | 151/2000 [00:41<03:48,  8.10it/s, est. speed input: 3472.58 toks/s, output: 116.16 toks/s][A
Processed prompts:   8%|▊         | 169/2000 [00:42<03:15,  9.38it/s, est. speed input: 3807.74 toks/s, output: 125.90 toks/s][A
Processed prompts:   9%|▉         | 186/2000 [00:44<02:58, 10.16it/s, est. speed input: 4098.51 toks/s, output: 134.28 toks/s][A
Processed prompts:  10%|█         | 202/2000 [00:45<02:52, 10.42it/s, est. speed input: 4328.63 toks/s, output: 141.29 toks/s][A
Processed prompts:  11%|█         | 221/2000 [00:46<02:36, 11.35it/s, est. speed input: 4502.10 toks/s, output: 150.09 toks/s][A
Processed prompts:  12%|█▏        | 238/2000 [00:48<02:31, 11.59it/s, est. speed input: 4633.82 toks/s, output: 157.03 toks/s][A
Processed prompts:  13%|█▎        | 256/2000 [00:49<02:25, 11.99it/s, est. speed input: 4774.34 toks/s, output: 164.23 toks/s][A
Processed prompts:  13%|█▎        | 261/2000 [00:51<03:03,  9.45it/s, est. speed input: 4719.76 toks/s, output: 162.81 toks/s][A
Processed prompts:  14%|█▎        | 272/2000 [00:52<03:11,  9.01it/s, est. speed input: 4812.80 toks/s, output: 165.42 toks/s][A
Processed prompts:  15%|█▍        | 294/2000 [00:53<02:34, 11.03it/s, est. speed input: 5090.89 toks/s, output: 174.49 toks/s][A
Processed prompts:  16%|█▋        | 330/2000 [00:55<02:00, 13.84it/s, est. speed input: 5494.86 toks/s, output: 189.10 toks/s][A
Processed prompts:  18%|█▊        | 360/2000 [00:57<01:42, 16.05it/s, est. speed input: 5821.68 toks/s, output: 201.27 toks/s][A
Processed prompts:  19%|█▉        | 384/2000 [00:58<01:38, 16.47it/s, est. speed input: 6045.38 toks/s, output: 209.14 toks/s][A
Processed prompts:  20%|█▉        | 399/2000 [00:59<01:47, 14.87it/s, est. speed input: 6153.49 toks/s, output: 212.00 toks/s][A
Processed prompts:  21%|██        | 422/2000 [01:01<01:43, 15.29it/s, est. speed input: 6405.69 toks/s, output: 218.55 toks/s][A
Processed prompts:  22%|██▏       | 431/2000 [01:02<02:02, 12.76it/s, est. speed input: 6418.70 toks/s, output: 218.26 toks/s][A
Processed prompts:  22%|██▏       | 435/2000 [01:04<02:38,  9.87it/s, est. speed input: 6347.89 toks/s, output: 215.54 toks/s][A
Processed prompts:  22%|██▏       | 443/2000 [01:05<03:01,  8.56it/s, est. speed input: 6340.22 toks/s, output: 214.75 toks/s][A
Processed prompts:  23%|██▎       | 453/2000 [01:06<03:09,  8.17it/s, est. speed input: 6372.47 toks/s, output: 215.11 toks/s][A
Processed prompts:  24%|██▍       | 475/2000 [01:08<02:25, 10.47it/s, est. speed input: 6595.62 toks/s, output: 221.19 toks/s][A
Processed prompts:  24%|██▍       | 490/2000 [01:09<02:22, 10.58it/s, est. speed input: 6700.84 toks/s, output: 223.73 toks/s][A
Processed prompts:  25%|██▍       | 499/2000 [01:11<02:42,  9.25it/s, est. speed input: 6702.93 toks/s, output: 223.31 toks/s][A
Processed prompts:  25%|██▌       | 501/2000 [01:12<03:21,  7.42it/s, est. speed input: 6634.43 toks/s, output: 220.87 toks/s][A
Processed prompts:  25%|██▌       | 502/2000 [01:12<03:34,  6.99it/s, est. speed input: 6620.41 toks/s, output: 220.28 toks/s][A
Processed prompts:  26%|██▌       | 519/2000 [01:13<02:45,  8.93it/s, est. speed input: 6767.51 toks/s, output: 223.28 toks/s][A
Processed prompts:  27%|██▋       | 536/2000 [01:15<02:25, 10.07it/s, est. speed input: 6908.82 toks/s, output: 226.16 toks/s][A
Processed prompts:  28%|██▊       | 554/2000 [01:16<02:13, 10.87it/s, est. speed input: 7055.64 toks/s, output: 229.16 toks/s][A
Processed prompts:  28%|██▊       | 570/2000 [01:18<02:09, 11.07it/s, est. speed input: 7142.70 toks/s, output: 231.43 toks/s][A
Processed prompts:  29%|██▉       | 587/2000 [01:19<02:03, 11.44it/s, est. speed input: 7219.90 toks/s, output: 234.01 toks/s][A
Processed prompts:  30%|███       | 604/2000 [01:20<02:00, 11.57it/s, est. speed input: 7291.47 toks/s, output: 236.39 toks/s][A
Processed prompts:  31%|███       | 621/2000 [01:22<01:56, 11.81it/s, est. speed input: 7363.15 toks/s, output: 238.85 toks/s][A
Processed prompts:  32%|███▏      | 637/2000 [01:23<01:56, 11.75it/s, est. speed input: 7384.00 toks/s, output: 240.84 toks/s][A
Processed prompts:  33%|███▎      | 654/2000 [01:25<01:52, 11.92it/s, est. speed input: 7412.76 toks/s, output: 243.12 toks/s][A
Processed prompts:  34%|███▎      | 671/2000 [01:26<01:51, 11.89it/s, est. speed input: 7435.96 toks/s, output: 245.17 toks/s][A
Processed prompts:  34%|███▍      | 687/2000 [01:27<01:51, 11.78it/s, est. speed input: 7484.15 toks/s, output: 246.95 toks/s][A
Processed prompts:  35%|███▌      | 704/2000 [01:29<01:48, 11.92it/s, est. speed input: 7581.01 toks/s, output: 249.00 toks/s][A
Processed prompts:  36%|███▌      | 721/2000 [01:30<01:46, 12.02it/s, est. speed input: 7675.73 toks/s, output: 251.01 toks/s][A
Processed prompts:  37%|███▋      | 742/2000 [01:32<01:38, 12.78it/s, est. speed input: 7800.75 toks/s, output: 254.08 toks/s][A
Processed prompts:  38%|███▊      | 757/2000 [01:33<01:41, 12.19it/s, est. speed input: 7832.77 toks/s, output: 255.24 toks/s][A
Processed prompts:  39%|███▉      | 780/2000 [01:34<01:30, 13.50it/s, est. speed input: 7933.98 toks/s, output: 258.93 toks/s][A
Processed prompts:  39%|███▉      | 788/2000 [01:36<01:49, 11.10it/s, est. speed input: 7890.39 toks/s, output: 257.68 toks/s][A
Processed prompts:  40%|███▉      | 798/2000 [01:37<01:51, 10.78it/s, est. speed input: 7898.79 toks/s, output: 258.16 toks/s][A
Processed prompts:  40%|████      | 802/2000 [01:37<01:56, 10.24it/s, est. speed input: 7894.90 toks/s, output: 258.02 toks/s][A
Processed prompts:  40%|████      | 809/2000 [01:38<01:58, 10.07it/s, est. speed input: 7913.34 toks/s, output: 258.36 toks/s][A
Processed prompts:  42%|████▏     | 831/2000 [01:39<01:35, 12.30it/s, est. speed input: 8044.90 toks/s, output: 261.98 toks/s][A
Processed prompts:  43%|████▎     | 852/2000 [01:41<01:27, 13.17it/s, est. speed input: 8130.41 toks/s, output: 264.88 toks/s][A
Processed prompts:  44%|████▍     | 885/2000 [01:42<01:06, 16.73it/s, est. speed input: 8331.06 toks/s, output: 271.58 toks/s][A
Processed prompts:  45%|████▌     | 906/2000 [01:44<01:07, 16.27it/s, est. speed input: 8393.94 toks/s, output: 274.30 toks/s][A
Processed prompts:  46%|████▌     | 919/2000 [01:45<01:16, 14.12it/s, est. speed input: 8386.91 toks/s, output: 274.55 toks/s][A
Processed prompts:  46%|████▌     | 921/2000 [01:46<01:47, 10.04it/s, est. speed input: 8291.72 toks/s, output: 271.45 toks/s][A
Processed prompts:  46%|████▋     | 925/2000 [01:48<02:16,  7.88it/s, est. speed input: 8226.92 toks/s, output: 269.20 toks/s][A
Processed prompts:  47%|████▋     | 942/2000 [01:49<01:54,  9.21it/s, est. speed input: 8292.08 toks/s, output: 270.88 toks/s][A
Processed prompts:  48%|████▊     | 958/2000 [01:51<01:44,  9.98it/s, est. speed input: 8348.19 toks/s, output: 272.30 toks/s][A
Processed prompts:  49%|████▉     | 980/2000 [01:52<01:27, 11.59it/s, est. speed input: 8452.44 toks/s, output: 275.17 toks/s][A
Processed prompts:  50%|█████     | 1001/2000 [01:53<01:18, 12.66it/s, est. speed input: 8547.62 toks/s, output: 277.79 toks/s][A
Processed prompts:  50%|█████     | 1003/2000 [01:54<01:39, 10.04it/s, est. speed input: 8488.27 toks/s, output: 275.82 toks/s][A
Processed prompts:  51%|█████     | 1018/2000 [01:56<01:34, 10.38it/s, est. speed input: 8527.03 toks/s, output: 276.75 toks/s][A
Processed prompts:  52%|█████▏    | 1043/2000 [01:57<01:15, 12.63it/s, est. speed input: 8651.22 toks/s, output: 280.10 toks/s][A
Processed prompts:  53%|█████▎    | 1063/2000 [01:59<01:11, 13.19it/s, est. speed input: 8733.04 toks/s, output: 282.12 toks/s][A
Processed prompts:  54%|█████▍    | 1081/2000 [02:00<01:09, 13.15it/s, est. speed input: 8796.39 toks/s, output: 283.56 toks/s][A
Processed prompts:  54%|█████▍    | 1088/2000 [02:01<01:25, 10.63it/s, est. speed input: 8757.40 toks/s, output: 282.09 toks/s][A
Processed prompts:  55%|█████▌    | 1101/2000 [02:03<01:25, 10.57it/s, est. speed input: 8781.03 toks/s, output: 282.61 toks/s][A
Processed prompts:  56%|█████▌    | 1118/2000 [02:04<01:19, 11.13it/s, est. speed input: 8785.99 toks/s, output: 283.87 toks/s][A
Processed prompts:  57%|█████▋    | 1135/2000 [02:05<01:15, 11.49it/s, est. speed input: 8790.14 toks/s, output: 285.08 toks/s][A
Processed prompts:  58%|█████▊    | 1159/2000 [02:07<01:03, 13.14it/s, est. speed input: 8847.91 toks/s, output: 287.86 toks/s][A
Processed prompts:  59%|█████▉    | 1185/2000 [02:08<00:54, 14.87it/s, est. speed input: 8944.21 toks/s, output: 291.10 toks/s][A
Processed prompts:  60%|██████    | 1203/2000 [02:10<00:55, 14.32it/s, est. speed input: 8992.51 toks/s, output: 292.30 toks/s][A
Processed prompts:  61%|██████    | 1218/2000 [02:11<00:59, 13.19it/s, est. speed input: 9014.02 toks/s, output: 292.71 toks/s][A
Processed prompts:  61%|██████    | 1221/2000 [02:12<01:18,  9.92it/s, est. speed input: 8944.69 toks/s, output: 290.40 toks/s][A
Processed prompts:  62%|██████▏   | 1236/2000 [02:14<01:14, 10.21it/s, est. speed input: 8971.70 toks/s, output: 290.99 toks/s][A
Processed prompts:  63%|██████▎   | 1252/2000 [02:15<01:10, 10.64it/s, est. speed input: 9006.44 toks/s, output: 291.83 toks/s][A
Processed prompts:  64%|██████▎   | 1270/2000 [02:17<01:04, 11.23it/s, est. speed input: 9048.86 toks/s, output: 292.97 toks/s][A
Processed prompts:  64%|██████▍   | 1285/2000 [02:18<01:04, 11.13it/s, est. speed input: 9061.41 toks/s, output: 293.53 toks/s][A
Processed prompts:  65%|██████▍   | 1296/2000 [02:19<01:03, 11.11it/s, est. speed input: 9045.98 toks/s, output: 293.96 toks/s][A
Processed prompts:  65%|██████▌   | 1303/2000 [02:20<01:04, 10.83it/s, est. speed input: 9030.34 toks/s, output: 294.04 toks/s][A
Processed prompts:  66%|██████▌   | 1312/2000 [02:21<01:03, 10.78it/s, est. speed input: 9015.90 toks/s, output: 294.30 toks/s][A
Processed prompts:  66%|██████▋   | 1328/2000 [02:22<01:00, 11.10it/s, est. speed input: 8999.17 toks/s, output: 295.04 toks/s][A
Processed prompts:  67%|██████▋   | 1341/2000 [02:23<00:58, 11.29it/s, est. speed input: 8998.43 toks/s, output: 295.65 toks/s][A
Processed prompts:  68%|██████▊   | 1361/2000 [02:24<00:51, 12.44it/s, est. speed input: 9051.59 toks/s, output: 297.29 toks/s][A
Processed prompts:  69%|██████▉   | 1379/2000 [02:26<00:49, 12.67it/s, est. speed input: 9097.67 toks/s, output: 298.43 toks/s][A
Processed prompts:  70%|██████▉   | 1392/2000 [02:27<00:52, 11.48it/s, est. speed input: 9103.20 toks/s, output: 298.35 toks/s][A
Processed prompts:  70%|███████   | 1408/2000 [02:29<00:51, 11.51it/s, est. speed input: 9134.73 toks/s, output: 299.01 toks/s][A
Processed prompts:  71%|███████▏  | 1426/2000 [02:30<00:48, 11.96it/s, est. speed input: 9180.77 toks/s, output: 300.07 toks/s][A
Processed prompts:  72%|███████▏  | 1441/2000 [02:31<00:47, 11.71it/s, est. speed input: 9207.50 toks/s, output: 300.57 toks/s][A
Processed prompts:  73%|███████▎  | 1458/2000 [02:33<00:46, 11.68it/s, est. speed input: 9240.91 toks/s, output: 301.25 toks/s][A
Processed prompts:  74%|███████▎  | 1470/2000 [02:34<00:47, 11.25it/s, est. speed input: 9252.96 toks/s, output: 301.43 toks/s][A
Processed prompts:  74%|███████▍  | 1485/2000 [02:35<00:45, 11.40it/s, est. speed input: 9279.38 toks/s, output: 302.06 toks/s][A
Processed prompts:  75%|███████▍  | 1499/2000 [02:36<00:43, 11.51it/s, est. speed input: 9304.26 toks/s, output: 302.66 toks/s][A
Processed prompts:  76%|███████▌  | 1515/2000 [02:37<00:32, 14.95it/s, est. speed input: 9389.39 toks/s, output: 305.19 toks/s][A
Processed prompts:  77%|███████▋  | 1531/2000 [02:37<00:22, 20.77it/s, est. speed input: 9489.06 toks/s, output: 308.24 toks/s][A
Processed prompts:  78%|███████▊  | 1551/2000 [02:37<00:14, 30.34it/s, est. speed input: 9613.34 toks/s, output: 312.08 toks/s][A
Processed prompts:  78%|███████▊  | 1568/2000 [02:37<00:10, 40.15it/s, est. speed input: 9718.39 toks/s, output: 315.32 toks/s][A
Processed prompts:  79%|███████▉  | 1587/2000 [02:37<00:07, 53.89it/s, est. speed input: 9819.67 toks/s, output: 318.95 toks/s][A
Processed prompts:  80%|████████  | 1604/2000 [02:37<00:05, 67.18it/s, est. speed input: 9883.41 toks/s, output: 322.17 toks/s][A
Processed prompts:  82%|████████▏ | 1635/2000 [02:38<00:03, 91.63it/s, est. speed input: 10000.35 toks/s, output: 328.08 toks/s][A
Processed prompts:  83%|████████▎ | 1651/2000 [02:38<00:03, 91.55it/s, est. speed input: 10084.19 toks/s, output: 331.02 toks/s][A
Processed prompts:  84%|████████▍ | 1684/2000 [02:38<00:02, 119.05it/s, est. speed input: 10295.35 toks/s, output: 337.56 toks/s][A
Processed prompts:  87%|████████▋ | 1733/2000 [02:38<00:01, 169.12it/s, est. speed input: 10547.75 toks/s, output: 347.24 toks/s][A
Processed prompts:  88%|████████▊ | 1765/2000 [02:38<00:01, 186.75it/s, est. speed input: 10700.31 toks/s, output: 353.42 toks/s][A
Processed prompts:  90%|████████▉ | 1796/2000 [02:38<00:00, 205.35it/s, est. speed input: 10892.61 toks/s, output: 359.42 toks/s][A
Processed prompts:  91%|█████████ | 1820/2000 [02:38<00:00, 207.82it/s, est. speed input: 11041.51 toks/s, output: 364.00 toks/s][A
Processed prompts:  92%|█████████▏| 1843/2000 [02:38<00:00, 212.47it/s, est. speed input: 11126.83 toks/s, output: 368.38 toks/s][A
Processed prompts:  94%|█████████▍| 1890/2000 [02:39<00:00, 257.53it/s, est. speed input: 11333.39 toks/s, output: 377.53 toks/s][A
Processed prompts:  97%|█████████▋| 1942/2000 [02:39<00:00, 318.00it/s, est. speed input: 11675.86 toks/s, output: 387.72 toks/s][A
Processed prompts:  99%|█████████▉| 1988/2000 [02:39<00:00, 342.33it/s, est. speed input: 12033.14 toks/s, output: 396.75 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:41<00:00, 12.35it/s, est. speed input: 11929.95 toks/s, output: 396.34 toks/s] 
2025-07-03 00:20:04,536 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 241 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\n\t\t\t\t\t\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-03 00:20:04,539 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 65 column 20 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\t\t\t\t\t\t\t', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-03 00:20:04,539 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 45 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\n\t    \t\t\t\t\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-03 00:20:04,540 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 65 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\t\t\t\t\t  \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-03 00:20:04,541 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 65 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\t\t\t\t\t  \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-03 00:20:04,986 - INFO - 
Progress: 18000 items processed
2025-07-03 00:20:04,987 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:20:04,987 - INFO -   accuracy: 0.689
2025-07-03 00:20:04,987 - INFO -   macro_f1: 0.503
2025-07-03 00:20:04,987 - INFO -   weighted_f1: 0.606
2025-07-03 00:20:04,987 - INFO - 
OVERALL metrics:
2025-07-03 00:20:04,987 - INFO -   exact_match_ratio: 0.660
2025-07-03 00:20:04,987 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:20:04,987 - INFO -   accuracy: 0.680
2025-07-03 00:20:04,987 - INFO -   macro_f1: 0.234
2025-07-03 00:20:04,987 - INFO -   weighted_f1: 0.589
2025-07-03 00:20:04,987 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:20:04,987 - INFO -   accuracy: 0.664
2025-07-03 00:20:04,987 - INFO -   macro_f1: 0.189
2025-07-03 00:20:04,987 - INFO -   weighted_f1: 0.562
 75%|███████▌  | 9/12 [25:20<08:24, 168.09s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:28<15:43:37, 28.32s/it, est. speed input: 37.71 toks/s, output: 1.09 toks/s][A
Processed prompts:   0%|          | 3/2000 [00:28<4:09:26,  7.49s/it, est. speed input: 111.31 toks/s, output: 3.24 toks/s][A
Processed prompts:   1%|          | 20/2000 [00:30<27:09,  1.21it/s, est. speed input: 706.63 toks/s, output: 20.59 toks/s][A
Processed prompts:   2%|▏         | 38/2000 [00:31<12:50,  2.55it/s, est. speed input: 1283.19 toks/s, output: 37.40 toks/s][A
Processed prompts:   3%|▎         | 56/2000 [00:32<08:09,  3.97it/s, est. speed input: 1808.85 toks/s, output: 52.69 toks/s][A
Processed prompts:   4%|▎         | 73/2000 [00:34<06:01,  5.34it/s, est. speed input: 2285.74 toks/s, output: 65.91 toks/s][A
Processed prompts:   5%|▍         | 91/2000 [00:35<04:40,  6.80it/s, est. speed input: 2761.39 toks/s, output: 78.98 toks/s][A
Processed prompts:   5%|▌         | 109/2000 [00:37<03:55,  8.04it/s, est. speed input: 3197.11 toks/s, output: 90.91 toks/s][A
Processed prompts:   6%|▋         | 126/2000 [00:38<03:27,  9.02it/s, est. speed input: 3563.80 toks/s, output: 101.31 toks/s][A
Processed prompts:   7%|▋         | 144/2000 [00:39<03:05, 10.01it/s, est. speed input: 3905.88 toks/s, output: 111.77 toks/s][A
Processed prompts:   8%|▊         | 162/2000 [00:41<02:50, 10.77it/s, est. speed input: 4225.33 toks/s, output: 121.49 toks/s][A
Processed prompts:   9%|▉         | 180/2000 [00:42<02:41, 11.25it/s, est. speed input: 4518.57 toks/s, output: 130.44 toks/s][A
Processed prompts:   9%|▉         | 182/2000 [00:43<03:21,  9.03it/s, est. speed input: 4438.58 toks/s, output: 128.61 toks/s][A
Processed prompts:  10%|▉         | 197/2000 [00:45<03:07,  9.63it/s, est. speed input: 4552.48 toks/s, output: 135.38 toks/s][A
Processed prompts:  11%|█         | 213/2000 [00:46<02:54, 10.26it/s, est. speed input: 4675.02 toks/s, output: 142.44 toks/s][A
Processed prompts:  12%|█▏        | 233/2000 [00:48<02:34, 11.43it/s, est. speed input: 4860.23 toks/s, output: 151.53 toks/s][A
Processed prompts:  12%|█▎        | 250/2000 [00:49<02:29, 11.70it/s, est. speed input: 5038.60 toks/s, output: 158.23 toks/s][A
Processed prompts:  14%|█▎        | 270/2000 [00:50<02:17, 12.55it/s, est. speed input: 5315.38 toks/s, output: 166.52 toks/s][A
Processed prompts:  15%|█▍        | 293/2000 [00:52<02:05, 13.65it/s, est. speed input: 5640.05 toks/s, output: 175.72 toks/s][A
Processed prompts:  16%|█▌        | 318/2000 [00:53<01:52, 14.92it/s, est. speed input: 5988.99 toks/s, output: 185.72 toks/s][A
Processed prompts:  17%|█▋        | 337/2000 [00:54<01:54, 14.58it/s, est. speed input: 6206.42 toks/s, output: 191.70 toks/s][A
Processed prompts:  17%|█▋        | 345/2000 [00:56<02:18, 11.96it/s, est. speed input: 6205.20 toks/s, output: 191.51 toks/s][A
Processed prompts:  18%|█▊        | 358/2000 [00:57<02:28, 11.05it/s, est. speed input: 6256.78 toks/s, output: 193.83 toks/s][A
Processed prompts:  19%|█▉        | 383/2000 [00:59<02:03, 13.12it/s, est. speed input: 6467.40 toks/s, output: 202.53 toks/s][A
Processed prompts:  20%|██        | 401/2000 [01:00<02:02, 13.06it/s, est. speed input: 6539.62 toks/s, output: 207.09 toks/s][A
Processed prompts:  21%|██        | 416/2000 [01:01<02:08, 12.35it/s, est. speed input: 6572.21 toks/s, output: 209.93 toks/s][A
Processed prompts:  22%|██▏       | 433/2000 [01:03<02:08, 12.24it/s, est. speed input: 6690.71 toks/s, output: 213.59 toks/s][A
Processed prompts:  23%|██▎       | 451/2000 [01:04<02:04, 12.46it/s, est. speed input: 6842.44 toks/s, output: 217.63 toks/s][A
Processed prompts:  23%|██▎       | 468/2000 [01:06<02:03, 12.45it/s, est. speed input: 6974.35 toks/s, output: 221.09 toks/s][A
Processed prompts:  24%|██▍       | 481/2000 [01:07<02:12, 11.49it/s, est. speed input: 7034.13 toks/s, output: 222.49 toks/s][A
Processed prompts:  24%|██▍       | 484/2000 [01:08<02:53,  8.75it/s, est. speed input: 6940.13 toks/s, output: 219.49 toks/s][A
Processed prompts:  25%|██▌       | 501/2000 [01:10<02:44,  9.09it/s, est. speed input: 7017.56 toks/s, output: 221.76 toks/s][A
Processed prompts:  25%|██▌       | 503/2000 [01:11<02:52,  8.66it/s, est. speed input: 7006.35 toks/s, output: 221.39 toks/s][A
Processed prompts:  26%|██▌       | 520/2000 [01:12<02:29,  9.90it/s, est. speed input: 7116.23 toks/s, output: 224.69 toks/s][A
Processed prompts:  27%|██▋       | 537/2000 [01:13<02:18, 10.56it/s, est. speed input: 7216.85 toks/s, output: 227.71 toks/s][A
Processed prompts:  28%|██▊       | 552/2000 [01:15<02:13, 10.84it/s, est. speed input: 7302.98 toks/s, output: 230.13 toks/s][A
Processed prompts:  28%|██▊       | 569/2000 [01:16<02:06, 11.30it/s, est. speed input: 7407.21 toks/s, output: 233.09 toks/s][A
Processed prompts:  29%|██▉       | 587/2000 [01:17<02:00, 11.70it/s, est. speed input: 7516.85 toks/s, output: 236.18 toks/s][A
Processed prompts:  31%|███       | 619/2000 [01:19<01:30, 15.22it/s, est. speed input: 7815.65 toks/s, output: 244.77 toks/s][A
Processed prompts:  32%|███▏      | 636/2000 [01:20<01:35, 14.35it/s, est. speed input: 7906.15 toks/s, output: 247.12 toks/s][A
Processed prompts:  33%|███▎      | 653/2000 [01:22<01:38, 13.71it/s, est. speed input: 7993.26 toks/s, output: 249.37 toks/s][A
Processed prompts:  33%|███▎      | 663/2000 [01:23<01:54, 11.65it/s, est. speed input: 7984.52 toks/s, output: 248.84 toks/s][A
Processed prompts:  34%|███▍      | 675/2000 [01:24<02:02, 10.77it/s, est. speed input: 8005.20 toks/s, output: 249.24 toks/s][A
Processed prompts:  34%|███▍      | 687/2000 [01:26<02:09, 10.11it/s, est. speed input: 8023.24 toks/s, output: 249.58 toks/s][A
Processed prompts:  35%|███▌      | 707/2000 [01:27<01:53, 11.36it/s, est. speed input: 8137.43 toks/s, output: 252.77 toks/s][A
Processed prompts:  36%|███▌      | 715/2000 [01:28<02:05, 10.26it/s, est. speed input: 8130.10 toks/s, output: 252.42 toks/s][A
Processed prompts:  37%|███▋      | 731/2000 [01:30<01:56, 10.88it/s, est. speed input: 8150.53 toks/s, output: 254.43 toks/s][A
Processed prompts:  38%|███▊      | 751/2000 [01:31<01:43, 12.03it/s, est. speed input: 8167.41 toks/s, output: 257.57 toks/s][A
Processed prompts:  38%|███▊      | 765/2000 [01:32<01:48, 11.35it/s, est. speed input: 8139.01 toks/s, output: 258.45 toks/s][A
Processed prompts:  39%|███▉      | 781/2000 [01:34<01:45, 11.51it/s, est. speed input: 8135.50 toks/s, output: 260.19 toks/s][A
Processed prompts:  39%|███▉      | 786/2000 [01:34<01:49, 11.05it/s, est. speed input: 8141.39 toks/s, output: 260.28 toks/s][A
Processed prompts:  40%|████      | 800/2000 [01:36<01:46, 11.31it/s, est. speed input: 8196.62 toks/s, output: 261.87 toks/s][A
Processed prompts:  41%|████      | 824/2000 [01:37<01:27, 13.43it/s, est. speed input: 8343.79 toks/s, output: 266.22 toks/s][A
Processed prompts:  42%|████▏     | 840/2000 [01:38<01:31, 12.67it/s, est. speed input: 8393.96 toks/s, output: 267.62 toks/s][A
Processed prompts:  43%|████▎     | 864/2000 [01:40<01:19, 14.22it/s, est. speed input: 8528.80 toks/s, output: 271.69 toks/s][A
Processed prompts:  44%|████▍     | 885/2000 [01:41<01:16, 14.62it/s, est. speed input: 8617.60 toks/s, output: 274.60 toks/s][A
Processed prompts:  46%|████▌     | 912/2000 [01:42<01:07, 16.02it/s, est. speed input: 8745.76 toks/s, output: 279.11 toks/s][A
Processed prompts:  47%|████▋     | 933/2000 [01:44<01:07, 15.83it/s, est. speed input: 8783.35 toks/s, output: 281.71 toks/s][A
Processed prompts:  47%|████▋     | 948/2000 [01:45<01:13, 14.35it/s, est. speed input: 8775.22 toks/s, output: 282.47 toks/s][A
Processed prompts:  48%|████▊     | 956/2000 [01:47<01:28, 11.75it/s, est. speed input: 8717.45 toks/s, output: 281.16 toks/s][A
Processed prompts:  48%|████▊     | 966/2000 [01:48<01:39, 10.34it/s, est. speed input: 8669.84 toks/s, output: 280.47 toks/s][A
Processed prompts:  49%|████▉     | 984/2000 [01:49<01:30, 11.20it/s, est. speed input: 8673.96 toks/s, output: 282.22 toks/s][A
Processed prompts:  50%|████▉     | 999/2000 [01:51<01:30, 11.12it/s, est. speed input: 8658.58 toks/s, output: 283.05 toks/s][A
Processed prompts:  50%|█████     | 1001/2000 [01:52<02:01,  8.22it/s, est. speed input: 8565.13 toks/s, output: 280.17 toks/s][A
Processed prompts:  50%|█████     | 1007/2000 [01:53<02:02,  8.10it/s, est. speed input: 8547.97 toks/s, output: 279.86 toks/s][A
Processed prompts:  52%|█████▏    | 1034/2000 [01:54<01:21, 11.89it/s, est. speed input: 8645.59 toks/s, output: 283.90 toks/s][A
Processed prompts:  53%|█████▎    | 1052/2000 [01:56<01:17, 12.29it/s, est. speed input: 8700.77 toks/s, output: 285.37 toks/s][A
Processed prompts:  53%|█████▎    | 1069/2000 [01:57<01:15, 12.33it/s, est. speed input: 8749.00 toks/s, output: 286.53 toks/s][A
Processed prompts:  54%|█████▍    | 1084/2000 [01:58<01:18, 11.73it/s, est. speed input: 8775.08 toks/s, output: 286.98 toks/s][A
Processed prompts:  55%|█████▌    | 1101/2000 [02:00<01:15, 11.93it/s, est. speed input: 8825.32 toks/s, output: 288.09 toks/s][A
Processed prompts:  56%|█████▌    | 1118/2000 [02:01<01:13, 12.05it/s, est. speed input: 8873.69 toks/s, output: 289.15 toks/s][A
Processed prompts:  57%|█████▋    | 1135/2000 [02:03<01:11, 12.05it/s, est. speed input: 8919.31 toks/s, output: 290.12 toks/s][A
Processed prompts:  58%|█████▊    | 1151/2000 [02:04<01:10, 11.97it/s, est. speed input: 8959.04 toks/s, output: 290.93 toks/s][A
Processed prompts:  58%|█████▊    | 1168/2000 [02:05<01:08, 12.10it/s, est. speed input: 9004.73 toks/s, output: 291.95 toks/s][A
Processed prompts:  59%|█████▉    | 1185/2000 [02:07<01:06, 12.20it/s, est. speed input: 9050.17 toks/s, output: 292.96 toks/s][A
Processed prompts:  60%|██████    | 1201/2000 [02:09<01:21,  9.81it/s, est. speed input: 9009.91 toks/s, output: 291.35 toks/s][A
Processed prompts:  61%|██████    | 1215/2000 [02:10<01:16, 10.25it/s, est. speed input: 8994.15 toks/s, output: 292.10 toks/s][A
Processed prompts:  62%|██████▏   | 1234/2000 [02:12<01:08, 11.21it/s, est. speed input: 8993.66 toks/s, output: 293.59 toks/s][A
Processed prompts:  63%|██████▎   | 1264/2000 [02:13<00:52, 13.99it/s, est. speed input: 9053.74 toks/s, output: 297.39 toks/s][A
Processed prompts:  64%|██████▍   | 1281/2000 [02:15<00:53, 13.49it/s, est. speed input: 9045.88 toks/s, output: 298.30 toks/s][A
Processed prompts:  65%|██████▍   | 1293/2000 [02:16<00:58, 12.08it/s, est. speed input: 9024.21 toks/s, output: 297.96 toks/s][A
Processed prompts:  65%|██████▌   | 1305/2000 [02:17<01:02, 11.05it/s, est. speed input: 9002.88 toks/s, output: 297.58 toks/s][A
Processed prompts:  66%|██████▌   | 1311/2000 [02:19<01:13,  9.39it/s, est. speed input: 8958.34 toks/s, output: 296.32 toks/s][A
Processed prompts:  66%|██████▌   | 1318/2000 [02:19<01:12,  9.41it/s, est. speed input: 8951.38 toks/s, output: 296.36 toks/s][A
Processed prompts:  67%|██████▋   | 1332/2000 [02:20<01:06, 10.11it/s, est. speed input: 8977.57 toks/s, output: 297.02 toks/s][A
Processed prompts:  67%|██████▋   | 1349/2000 [02:22<01:00, 10.83it/s, est. speed input: 9017.05 toks/s, output: 297.96 toks/s][A
Processed prompts:  68%|██████▊   | 1365/2000 [02:23<00:58, 10.94it/s, est. speed input: 9045.48 toks/s, output: 298.55 toks/s][A
Processed prompts:  69%|██████▉   | 1382/2000 [02:25<00:54, 11.35it/s, est. speed input: 9083.60 toks/s, output: 299.44 toks/s][A
Processed prompts:  70%|██████▉   | 1397/2000 [02:26<00:52, 11.40it/s, est. speed input: 9111.90 toks/s, output: 300.07 toks/s][A
Processed prompts:  71%|███████   | 1416/2000 [02:27<00:48, 12.10it/s, est. speed input: 9162.70 toks/s, output: 301.36 toks/s][A
Processed prompts:  72%|███████▏  | 1431/2000 [02:29<00:49, 11.57it/s, est. speed input: 9181.36 toks/s, output: 301.67 toks/s][A
Processed prompts:  72%|███████▏  | 1439/2000 [02:30<00:51, 10.95it/s, est. speed input: 9182.31 toks/s, output: 301.56 toks/s][A
Processed prompts:  73%|███████▎  | 1452/2000 [02:31<00:49, 11.05it/s, est. speed input: 9178.79 toks/s, output: 302.08 toks/s][A
Processed prompts:  74%|███████▎  | 1472/2000 [02:32<00:43, 12.15it/s, est. speed input: 9192.32 toks/s, output: 303.61 toks/s][A
Processed prompts:  75%|███████▍  | 1493/2000 [02:34<00:39, 12.92it/s, est. speed input: 9207.35 toks/s, output: 305.15 toks/s][A
Processed prompts:  76%|███████▌  | 1521/2000 [02:35<00:30, 15.48it/s, est. speed input: 9264.04 toks/s, output: 308.35 toks/s][A
Processed prompts:  77%|███████▋  | 1539/2000 [02:35<00:22, 20.58it/s, est. speed input: 9343.47 toks/s, output: 311.72 toks/s][A
Processed prompts:  78%|███████▊  | 1557/2000 [02:35<00:16, 27.39it/s, est. speed input: 9423.32 toks/s, output: 315.09 toks/s][A
Processed prompts:  78%|███████▊  | 1567/2000 [02:35<00:14, 29.61it/s, est. speed input: 9476.22 toks/s, output: 316.68 toks/s][A
Processed prompts:  79%|███████▉  | 1586/2000 [02:36<00:10, 40.89it/s, est. speed input: 9602.17 toks/s, output: 320.37 toks/s][A
Processed prompts:  80%|████████  | 1608/2000 [02:36<00:06, 57.14it/s, est. speed input: 9729.24 toks/s, output: 324.58 toks/s][A
Processed prompts:  83%|████████▎ | 1661/2000 [02:36<00:03, 103.01it/s, est. speed input: 10037.90 toks/s, output: 334.97 toks/s][A
Processed prompts:  84%|████████▍ | 1680/2000 [02:36<00:03, 105.02it/s, est. speed input: 10117.44 toks/s, output: 338.45 toks/s][A
Processed prompts:  85%|████████▌ | 1701/2000 [02:36<00:02, 110.01it/s, est. speed input: 10268.83 toks/s, output: 342.39 toks/s][A
Processed prompts:  86%|████████▋ | 1727/2000 [02:36<00:02, 122.63it/s, est. speed input: 10458.79 toks/s, output: 347.35 toks/s][A
Processed prompts:  88%|████████▊ | 1764/2000 [02:37<00:01, 152.44it/s, est. speed input: 10733.72 toks/s, output: 354.60 toks/s][A
Processed prompts:  90%|█████████ | 1807/2000 [02:37<00:00, 198.57it/s, est. speed input: 11030.26 toks/s, output: 363.03 toks/s][A
Processed prompts:  92%|█████████▏| 1832/2000 [02:37<00:00, 208.42it/s, est. speed input: 11155.44 toks/s, output: 367.75 toks/s][A
Processed prompts:  93%|█████████▎| 1864/2000 [02:37<00:00, 212.18it/s, est. speed input: 11311.85 toks/s, output: 373.74 toks/s][A
Processed prompts:  96%|█████████▌| 1913/2000 [02:37<00:00, 267.54it/s, est. speed input: 11634.88 toks/s, output: 383.43 toks/s][A
Processed prompts:  98%|█████████▊| 1966/2000 [02:37<00:00, 309.03it/s, est. speed input: 11975.07 toks/s, output: 393.96 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:37<00:00, 12.69it/s, est. speed input: 12170.28 toks/s, output: 400.78 toks/s] 
2025-07-03 00:22:53,385 - INFO - 
Progress: 20000 items processed
2025-07-03 00:22:53,385 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:22:53,385 - INFO -   accuracy: 0.634
2025-07-03 00:22:53,385 - INFO -   macro_f1: 0.486
2025-07-03 00:22:53,385 - INFO -   weighted_f1: 0.543
2025-07-03 00:22:53,385 - INFO - 
OVERALL metrics:
2025-07-03 00:22:53,386 - INFO -   exact_match_ratio: 0.598
2025-07-03 00:22:53,386 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:22:53,386 - INFO -   accuracy: 0.620
2025-07-03 00:22:53,386 - INFO -   macro_f1: 0.220
2025-07-03 00:22:53,386 - INFO -   weighted_f1: 0.517
2025-07-03 00:22:53,386 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:22:53,386 - INFO -   accuracy: 0.609
2025-07-03 00:22:53,386 - INFO -   macro_f1: 0.195
2025-07-03 00:22:53,386 - INFO -   weighted_f1: 0.494
 83%|████████▎ | 10/12 [28:08<05:36, 168.19s/it]
Processed prompts:   0%|          | 0/2000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/2000 [00:28<15:48:22, 28.47s/it, est. speed input: 26.14 toks/s, output: 1.09 toks/s][A
Processed prompts:   0%|          | 2/2000 [00:28<6:36:43, 11.91s/it, est. speed input: 51.54 toks/s, output: 2.15 toks/s] [A
Processed prompts:   1%|          | 16/2000 [00:29<33:25,  1.01s/it, est. speed input: 396.99 toks/s, output: 16.58 toks/s][A
Processed prompts:   2%|▏         | 33/2000 [00:31<14:27,  2.27it/s, est. speed input: 782.15 toks/s, output: 32.66 toks/s][A
Processed prompts:   2%|▏         | 49/2000 [00:32<09:06,  3.57it/s, est. speed input: 1211.40 toks/s, output: 46.48 toks/s][A
Processed prompts:   4%|▎         | 71/2000 [00:34<05:43,  5.62it/s, est. speed input: 1810.26 toks/s, output: 64.62 toks/s][A
Processed prompts:   4%|▍         | 87/2000 [00:35<04:44,  6.71it/s, est. speed input: 2202.39 toks/s, output: 75.95 toks/s][A
Processed prompts:   5%|▌         | 104/2000 [00:36<03:59,  7.91it/s, est. speed input: 2556.77 toks/s, output: 87.41 toks/s][A
Processed prompts:   6%|▌         | 124/2000 [00:38<03:18,  9.45it/s, est. speed input: 2860.01 toks/s, output: 100.49 toks/s][A
Processed prompts:   7%|▋         | 142/2000 [00:39<02:59, 10.34it/s, est. speed input: 3097.71 toks/s, output: 111.02 toks/s][A
Processed prompts:   8%|▊         | 159/2000 [00:41<02:51, 10.70it/s, est. speed input: 3295.52 toks/s, output: 119.88 toks/s][A
Processed prompts:   9%|▉         | 176/2000 [00:42<02:43, 11.13it/s, est. speed input: 3485.50 toks/s, output: 128.37 toks/s][A
Processed prompts:  10%|▉         | 194/2000 [00:43<02:34, 11.65it/s, est. speed input: 3680.11 toks/s, output: 137.01 toks/s][A
Processed prompts:  11%|█         | 212/2000 [00:45<02:28, 12.04it/s, est. speed input: 3863.21 toks/s, output: 145.13 toks/s][A
Processed prompts:  11%|█▏        | 228/2000 [00:46<02:31, 11.73it/s, est. speed input: 4053.36 toks/s, output: 151.22 toks/s][A
Processed prompts:  12%|█▏        | 245/2000 [00:48<02:27, 11.86it/s, est. speed input: 4309.94 toks/s, output: 157.77 toks/s][A
Processed prompts:  13%|█▎        | 263/2000 [00:49<02:22, 12.15it/s, est. speed input: 4574.46 toks/s, output: 164.56 toks/s][A
Processed prompts:  14%|█▍        | 288/2000 [00:51<02:05, 13.68it/s, est. speed input: 4964.72 toks/s, output: 174.89 toks/s][A
Processed prompts:  15%|█▍        | 298/2000 [00:52<02:24, 11.78it/s, est. speed input: 5035.66 toks/s, output: 176.02 toks/s][A
Processed prompts:  16%|█▌        | 314/2000 [00:53<02:23, 11.71it/s, est. speed input: 5223.48 toks/s, output: 180.67 toks/s][A
Processed prompts:  16%|█▌        | 323/2000 [00:55<02:45, 10.15it/s, est. speed input: 5264.96 toks/s, output: 181.29 toks/s][A
Processed prompts:  17%|█▋        | 337/2000 [00:56<02:45, 10.07it/s, est. speed input: 5396.17 toks/s, output: 184.57 toks/s][A
Processed prompts:  18%|█▊        | 362/2000 [00:58<02:11, 12.50it/s, est. speed input: 5727.90 toks/s, output: 193.75 toks/s][A
Processed prompts:  19%|█▉        | 380/2000 [00:59<02:07, 12.70it/s, est. speed input: 5918.95 toks/s, output: 198.73 toks/s][A
Processed prompts:  20%|█▉        | 395/2000 [01:00<02:13, 12.02it/s, est. speed input: 6042.28 toks/s, output: 201.72 toks/s][A
Processed prompts:  20%|██        | 401/2000 [01:02<02:55,  9.11it/s, est. speed input: 5974.83 toks/s, output: 199.27 toks/s][A
Processed prompts:  21%|██        | 414/2000 [01:03<02:44,  9.66it/s, est. speed input: 6019.53 toks/s, output: 202.42 toks/s][A
Processed prompts:  22%|██▏       | 436/2000 [01:05<02:15, 11.52it/s, est. speed input: 6143.58 toks/s, output: 209.14 toks/s][A
Processed prompts:  24%|██▎       | 470/2000 [01:06<01:40, 15.23it/s, est. speed input: 6395.85 toks/s, output: 220.97 toks/s][A
Processed prompts:  25%|██▌       | 500/2000 [01:07<01:27, 17.17it/s, est. speed input: 6600.76 toks/s, output: 230.51 toks/s][A
Processed prompts:  25%|██▌       | 502/2000 [01:10<02:36,  9.58it/s, est. speed input: 6361.62 toks/s, output: 222.28 toks/s][A
Processed prompts:  26%|██▌       | 516/2000 [01:12<02:34,  9.59it/s, est. speed input: 6381.80 toks/s, output: 223.82 toks/s][A
Processed prompts:  26%|██▌       | 521/2000 [01:12<02:44,  9.00it/s, est. speed input: 6367.28 toks/s, output: 223.45 toks/s][A
Processed prompts:  27%|██▋       | 531/2000 [01:13<02:37,  9.32it/s, est. speed input: 6433.43 toks/s, output: 224.89 toks/s][A
Processed prompts:  27%|██▋       | 548/2000 [01:15<02:22, 10.21it/s, est. speed input: 6559.70 toks/s, output: 227.92 toks/s][A
Processed prompts:  28%|██▊       | 564/2000 [01:16<02:17, 10.45it/s, est. speed input: 6662.93 toks/s, output: 230.28 toks/s][A
Processed prompts:  29%|██▉       | 580/2000 [01:18<02:10, 10.85it/s, est. speed input: 6770.94 toks/s, output: 232.84 toks/s][A
Processed prompts:  30%|██▉       | 598/2000 [01:19<02:02, 11.45it/s, est. speed input: 6892.06 toks/s, output: 235.97 toks/s][A
Processed prompts:  31%|███       | 614/2000 [01:20<02:00, 11.53it/s, est. speed input: 6986.37 toks/s, output: 238.33 toks/s][A
Processed prompts:  32%|███▏      | 631/2000 [01:22<01:57, 11.64it/s, est. speed input: 7084.82 toks/s, output: 240.81 toks/s][A
Processed prompts:  32%|███▏      | 644/2000 [01:23<01:56, 11.62it/s, est. speed input: 7157.55 toks/s, output: 242.56 toks/s][A
Processed prompts:  33%|███▎      | 664/2000 [01:24<01:46, 12.50it/s, est. speed input: 7297.10 toks/s, output: 246.17 toks/s][A
Processed prompts:  34%|███▍      | 681/2000 [01:26<01:46, 12.43it/s, est. speed input: 7395.66 toks/s, output: 248.52 toks/s][A
Processed prompts:  35%|███▍      | 698/2000 [01:27<01:46, 12.22it/s, est. speed input: 7485.99 toks/s, output: 250.63 toks/s][A
Processed prompts:  35%|███▌      | 706/2000 [01:28<01:57, 10.98it/s, est. speed input: 7496.04 toks/s, output: 250.42 toks/s][A
Processed prompts:  36%|███▌      | 722/2000 [01:30<01:53, 11.23it/s, est. speed input: 7587.85 toks/s, output: 252.41 toks/s][A
Processed prompts:  37%|███▋      | 737/2000 [01:31<01:51, 11.37it/s, est. speed input: 7670.34 toks/s, output: 254.26 toks/s][A
Processed prompts:  38%|███▊      | 756/2000 [01:32<01:43, 12.02it/s, est. speed input: 7788.84 toks/s, output: 257.05 toks/s][A
Processed prompts:  39%|███▉      | 782/2000 [01:34<01:26, 14.15it/s, est. speed input: 7993.58 toks/s, output: 262.23 toks/s][A
Processed prompts:  40%|███▉      | 799/2000 [01:35<01:28, 13.58it/s, est. speed input: 8082.68 toks/s, output: 264.13 toks/s][A
Processed prompts:  41%|████      | 816/2000 [01:36<01:30, 13.15it/s, est. speed input: 8168.13 toks/s, output: 265.93 toks/s][A
Processed prompts:  41%|████▏     | 829/2000 [01:38<01:36, 12.08it/s, est. speed input: 8181.51 toks/s, output: 266.50 toks/s][A
Processed prompts:  42%|████▏     | 845/2000 [01:39<01:36, 11.96it/s, est. speed input: 8207.00 toks/s, output: 268.01 toks/s][A
Processed prompts:  43%|████▎     | 862/2000 [01:41<01:33, 12.11it/s, est. speed input: 8241.06 toks/s, output: 269.82 toks/s][A
Processed prompts:  44%|████▍     | 875/2000 [01:42<01:40, 11.23it/s, est. speed input: 8241.08 toks/s, output: 270.22 toks/s][A
Processed prompts:  45%|████▍     | 896/2000 [01:43<01:28, 12.49it/s, est. speed input: 8334.80 toks/s, output: 273.24 toks/s][A
Processed prompts:  45%|████▌     | 901/2000 [01:44<01:44, 10.48it/s, est. speed input: 8298.88 toks/s, output: 271.96 toks/s][A
Processed prompts:  46%|████▌     | 916/2000 [01:46<01:39, 10.89it/s, est. speed input: 8350.28 toks/s, output: 273.20 toks/s][A
Processed prompts:  46%|████▋     | 928/2000 [01:47<01:39, 10.81it/s, est. speed input: 8381.23 toks/s, output: 273.90 toks/s][A
Processed prompts:  47%|████▋     | 939/2000 [01:48<01:37, 10.89it/s, est. speed input: 8413.07 toks/s, output: 274.70 toks/s][A
Processed prompts:  48%|████▊     | 963/2000 [01:49<01:18, 13.13it/s, est. speed input: 8541.03 toks/s, output: 278.34 toks/s][A
Processed prompts:  49%|████▉     | 980/2000 [01:50<01:18, 12.92it/s, est. speed input: 8599.33 toks/s, output: 279.82 toks/s][A
Processed prompts:  50%|████▉     | 997/2000 [01:52<01:19, 12.68it/s, est. speed input: 8653.40 toks/s, output: 281.18 toks/s][A
Processed prompts:  51%|█████     | 1012/2000 [01:53<01:21, 12.19it/s, est. speed input: 8691.22 toks/s, output: 282.06 toks/s][A
Processed prompts:  51%|█████     | 1021/2000 [01:54<01:30, 10.76it/s, est. speed input: 8679.97 toks/s, output: 281.51 toks/s][A
Processed prompts:  52%|█████▏    | 1031/2000 [01:55<01:29, 10.80it/s, est. speed input: 8703.24 toks/s, output: 282.05 toks/s][A
Processed prompts:  52%|█████▏    | 1047/2000 [01:57<01:27, 10.95it/s, est. speed input: 8743.02 toks/s, output: 282.99 toks/s][A
Processed prompts:  53%|█████▎    | 1061/2000 [01:58<01:24, 11.12it/s, est. speed input: 8778.88 toks/s, output: 283.88 toks/s][A
Processed prompts:  53%|█████▎    | 1063/2000 [01:58<01:29, 10.44it/s, est. speed input: 8768.10 toks/s, output: 283.53 toks/s][A
Processed prompts:  54%|█████▍    | 1080/2000 [02:00<01:22, 11.19it/s, est. speed input: 8814.74 toks/s, output: 284.97 toks/s][A
Processed prompts:  55%|█████▌    | 1107/2000 [02:01<01:03, 14.13it/s, est. speed input: 8914.57 toks/s, output: 288.91 toks/s][A
Processed prompts:  57%|█████▋    | 1131/2000 [02:03<00:57, 15.20it/s, est. speed input: 8989.36 toks/s, output: 291.97 toks/s][A
Processed prompts:  58%|█████▊    | 1157/2000 [02:04<00:50, 16.53it/s, est. speed input: 9042.87 toks/s, output: 295.44 toks/s][A
Processed prompts:  59%|█████▊    | 1171/2000 [02:05<00:56, 14.63it/s, est. speed input: 9015.77 toks/s, output: 295.78 toks/s][A
Processed prompts:  59%|█████▉    | 1176/2000 [02:07<01:14, 11.04it/s, est. speed input: 8938.54 toks/s, output: 293.69 toks/s][A
Processed prompts:  59%|█████▉    | 1181/2000 [02:08<01:33,  8.76it/s, est. speed input: 8869.99 toks/s, output: 291.81 toks/s][A
Processed prompts:  60%|█████▉    | 1193/2000 [02:09<01:28,  9.17it/s, est. speed input: 8888.97 toks/s, output: 292.24 toks/s][A
Processed prompts:  60%|██████    | 1206/2000 [02:10<01:20,  9.86it/s, est. speed input: 8919.14 toks/s, output: 293.05 toks/s][A
Processed prompts:  62%|██████▏   | 1234/2000 [02:12<00:58, 13.08it/s, est. speed input: 9019.41 toks/s, output: 296.69 toks/s][A
Processed prompts:  63%|██████▎   | 1266/2000 [02:13<00:44, 16.34it/s, est. speed input: 9141.84 toks/s, output: 301.33 toks/s][A
Processed prompts:  64%|██████▍   | 1289/2000 [02:14<00:43, 16.53it/s, est. speed input: 9182.01 toks/s, output: 303.61 toks/s][A
Processed prompts:  66%|██████▌   | 1314/2000 [02:16<00:40, 17.11it/s, est. speed input: 9215.28 toks/s, output: 306.19 toks/s][A
Processed prompts:  66%|██████▋   | 1329/2000 [02:17<00:44, 15.16it/s, est. speed input: 9188.71 toks/s, output: 306.39 toks/s][A
Processed prompts:  67%|██████▋   | 1345/2000 [02:19<00:46, 14.18it/s, est. speed input: 9169.81 toks/s, output: 306.95 toks/s][A
Processed prompts:  68%|██████▊   | 1350/2000 [02:20<00:59, 11.01it/s, est. speed input: 9102.65 toks/s, output: 305.09 toks/s][A
Processed prompts:  68%|██████▊   | 1357/2000 [02:21<01:10,  9.13it/s, est. speed input: 9041.61 toks/s, output: 303.61 toks/s][A
Processed prompts:  69%|██████▊   | 1371/2000 [02:23<01:06,  9.42it/s, est. speed input: 9024.63 toks/s, output: 303.81 toks/s][A
Processed prompts:  69%|██████▉   | 1388/2000 [02:24<00:59, 10.30it/s, est. speed input: 9025.64 toks/s, output: 304.67 toks/s][A
Processed prompts:  70%|███████   | 1401/2000 [02:25<00:59, 10.04it/s, est. speed input: 9006.74 toks/s, output: 304.65 toks/s][A
Processed prompts:  71%|███████   | 1416/2000 [02:27<00:57, 10.16it/s, est. speed input: 8994.57 toks/s, output: 304.93 toks/s][A
Processed prompts:  71%|███████   | 1421/2000 [02:28<01:06,  8.72it/s, est. speed input: 8952.48 toks/s, output: 303.74 toks/s][A
Processed prompts:  71%|███████▏  | 1428/2000 [02:29<01:04,  8.82it/s, est. speed input: 8942.23 toks/s, output: 303.74 toks/s][A
Processed prompts:  72%|███████▏  | 1440/2000 [02:30<00:59,  9.47it/s, est. speed input: 8936.79 toks/s, output: 304.18 toks/s][A
Processed prompts:  73%|███████▎  | 1456/2000 [02:31<00:52, 10.29it/s, est. speed input: 8935.96 toks/s, output: 304.94 toks/s][A
Processed prompts:  74%|███████▍  | 1480/2000 [02:33<00:41, 12.57it/s, est. speed input: 8970.87 toks/s, output: 307.28 toks/s][A
Processed prompts:  75%|███████▌  | 1506/2000 [02:34<00:33, 14.61it/s, est. speed input: 9016.27 toks/s, output: 309.98 toks/s][A
Processed prompts:  76%|███████▌  | 1521/2000 [02:35<00:32, 14.83it/s, est. speed input: 9032.00 toks/s, output: 311.14 toks/s][A
Processed prompts:  77%|███████▋  | 1531/2000 [02:35<00:26, 17.70it/s, est. speed input: 9073.28 toks/s, output: 312.97 toks/s][A
Processed prompts:  77%|███████▋  | 1541/2000 [02:35<00:21, 21.41it/s, est. speed input: 9116.44 toks/s, output: 314.80 toks/s][A
Processed prompts:  77%|███████▋  | 1547/2000 [02:35<00:19, 23.58it/s, est. speed input: 9150.60 toks/s, output: 315.83 toks/s][A
Processed prompts:  78%|███████▊  | 1561/2000 [02:35<00:13, 32.94it/s, est. speed input: 9240.05 toks/s, output: 318.59 toks/s][A
Processed prompts:  78%|███████▊  | 1569/2000 [02:36<00:11, 37.61it/s, est. speed input: 9288.43 toks/s, output: 320.07 toks/s][A
Processed prompts:  79%|███████▉  | 1580/2000 [02:36<00:09, 46.47it/s, est. speed input: 9356.98 toks/s, output: 322.16 toks/s][A
Processed prompts:  80%|████████  | 1608/2000 [02:36<00:04, 82.31it/s, est. speed input: 9517.41 toks/s, output: 327.79 toks/s][A
Processed prompts:  83%|████████▎ | 1655/2000 [02:36<00:02, 133.04it/s, est. speed input: 9766.30 toks/s, output: 336.96 toks/s][A
Processed prompts:  85%|████████▍ | 1697/2000 [02:36<00:01, 168.15it/s, est. speed input: 10058.37 toks/s, output: 344.98 toks/s][A
Processed prompts:  86%|████████▌ | 1720/2000 [02:36<00:01, 162.84it/s, est. speed input: 10238.65 toks/s, output: 349.18 toks/s][A
Processed prompts:  87%|████████▋ | 1739/2000 [02:36<00:01, 151.10it/s, est. speed input: 10353.76 toks/s, output: 352.71 toks/s][A
Processed prompts:  89%|████████▉ | 1788/2000 [02:36<00:01, 209.68it/s, est. speed input: 10643.16 toks/s, output: 362.24 toks/s][A
Processed prompts:  91%|█████████▏| 1826/2000 [02:37<00:00, 242.95it/s, est. speed input: 10826.78 toks/s, output: 369.50 toks/s][A
Processed prompts:  93%|█████████▎| 1862/2000 [02:37<00:00, 245.48it/s, est. speed input: 11027.90 toks/s, output: 376.42 toks/s][A
Processed prompts:  96%|█████████▌| 1913/2000 [02:37<00:00, 299.17it/s, est. speed input: 11358.25 toks/s, output: 386.38 toks/s][A
Processed prompts:  98%|█████████▊| 1967/2000 [02:37<00:00, 336.60it/s, est. speed input: 11683.38 toks/s, output: 396.81 toks/s][AProcessed prompts: 100%|██████████| 2000/2000 [02:37<00:00, 12.70it/s, est. speed input: 11821.52 toks/s, output: 403.23 toks/s] 
2025-07-03 00:25:41,648 - INFO - 
Progress: 22000 items processed
2025-07-03 00:25:41,648 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:25:41,648 - INFO -   accuracy: 0.580
2025-07-03 00:25:41,648 - INFO -   macro_f1: 0.453
2025-07-03 00:25:41,648 - INFO -   weighted_f1: 0.478
2025-07-03 00:25:41,648 - INFO - 
OVERALL metrics:
2025-07-03 00:25:41,648 - INFO -   exact_match_ratio: 0.545
2025-07-03 00:25:41,648 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:25:41,648 - INFO -   accuracy: 0.566
2025-07-03 00:25:41,648 - INFO -   macro_f1: 0.202
2025-07-03 00:25:41,648 - INFO -   weighted_f1: 0.452
2025-07-03 00:25:41,648 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:25:41,648 - INFO -   accuracy: 0.557
2025-07-03 00:25:41,648 - INFO -   macro_f1: 0.187
2025-07-03 00:25:41,648 - INFO -   weighted_f1: 0.431
 92%|█████████▏| 11/12 [30:56<02:48, 168.21s/it]
Processed prompts:   0%|          | 0/80 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   1%|▏         | 1/80 [00:05<07:46,  5.90s/it, est. speed input: 105.79 toks/s, output: 5.09 toks/s][A
Processed prompts:  31%|███▏      | 25/80 [00:06<00:09,  5.80it/s, est. speed input: 2706.06 toks/s, output: 127.74 toks/s][A
Processed prompts:  86%|████████▋ | 69/80 [00:06<00:00, 19.86it/s, est. speed input: 7986.20 toks/s, output: 354.43 toks/s][AProcessed prompts: 100%|██████████| 80/80 [00:08<00:00,  9.12it/s, est. speed input: 6510.48 toks/s, output: 343.21 toks/s]
2025-07-03 00:25:50,713 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 123 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\n\t\t\t\t\t\t\t\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-03 00:25:50,714 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 499 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...n\n\n\n\n\n\n\n\n\n\n\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-03 00:25:50,715 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 65 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\t\t\t\t\t\t\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-03 00:25:50,715 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 65 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\t\t\t\t\t  \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-03 00:25:51,273 - INFO - 
Progress: 22080 items processed
2025-07-03 00:25:51,273 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:25:51,273 - INFO -   accuracy: 0.578
2025-07-03 00:25:51,273 - INFO -   macro_f1: 0.451
2025-07-03 00:25:51,273 - INFO -   weighted_f1: 0.475
2025-07-03 00:25:51,273 - INFO - 
OVERALL metrics:
2025-07-03 00:25:51,273 - INFO -   exact_match_ratio: 0.543
2025-07-03 00:25:51,273 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:25:51,273 - INFO -   accuracy: 0.564
2025-07-03 00:25:51,273 - INFO -   macro_f1: 0.201
2025-07-03 00:25:51,273 - INFO -   weighted_f1: 0.450
2025-07-03 00:25:51,273 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:25:51,274 - INFO -   accuracy: 0.555
2025-07-03 00:25:51,274 - INFO -   macro_f1: 0.187
2025-07-03 00:25:51,274 - INFO -   weighted_f1: 0.428
100%|██████████| 12/12 [31:06<00:00, 119.97s/it]100%|██████████| 12/12 [31:08<00:00, 155.72s/it]
2025-07-03 00:25:54,323 - INFO - 
Final Metrics:
2025-07-03 00:25:54,323 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-03 00:25:54,323 - INFO -   accuracy: 0.578
2025-07-03 00:25:54,323 - INFO -   macro_f1: 0.451
2025-07-03 00:25:54,323 - INFO -   weighted_f1: 0.475
2025-07-03 00:25:54,323 - INFO - 
OVERALL metrics:
2025-07-03 00:25:54,323 - INFO -   exact_match_ratio: 0.543
2025-07-03 00:25:54,323 - INFO - 
TARGET_GROUP metrics:
2025-07-03 00:25:54,323 - INFO -   accuracy: 0.564
2025-07-03 00:25:54,323 - INFO -   macro_f1: 0.201
2025-07-03 00:25:54,324 - INFO -   weighted_f1: 0.450
2025-07-03 00:25:54,324 - INFO - 
ATTACK_METHOD metrics:
2025-07-03 00:25:54,324 - INFO -   accuracy: 0.555
2025-07-03 00:25:54,324 - INFO -   macro_f1: 0.187
2025-07-03 00:25:54,324 - INFO -   weighted_f1: 0.428
2025-07-03 00:25:54,585 - INFO - Results saved to data/results/img_classification/Qwen2.5-VL-7B-Instruct/20250702_235302.json
2025-07-03 00:25:54,585 - INFO - Pipeline completed successfully!
