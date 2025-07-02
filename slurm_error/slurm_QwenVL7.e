2025-07-02 17:44:10,291 - INFO - ======================================================================
2025-07-02 17:44:10,292 - INFO - Extreme personas path: data/results/extreme_pos_personas/Qwen2.5-VL-7B-Instruct/extreme_pos_corners.pkl
2025-07-02 17:44:10,292 - INFO - Output path: data/results/img_classification/Qwen2.5-VL-7B-Instruct/20250702_174410.json
2025-07-02 17:44:10,292 - INFO - Using model: Qwen/Qwen2.5-VL-7B-Instruct
2025-07-02 17:44:10,292 - INFO - ======================================================================

2025-07-02 17:44:10,302 - INFO - ======================================================================
2025-07-02 17:44:10,303 - INFO - VLLM Parameters:
2025-07-02 17:44:10,303 - INFO - {'model': 'Qwen/Qwen2.5-VL-7B-Instruct', 'tokenizer_mode': 'auto', 'trust_remote_code': True, 'enforce_eager': False, 'dtype': 'auto', 'gpu_memory_utilization': 0.95, 'tensor_parallel_size': 1, 'enable_prefix_caching': True, 'disable_log_stats': True, 'download_dir': '/scratch/user/uqscivel/HF-CACHE', 'limit_mm_per_prompt': {'image': 1}, 'max_model_len': 2048, 'max_num_seqs': 500}
2025-07-02 17:44:10,303 - INFO - ======================================================================

Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:01<00:05,  1.48s/it]
Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:06<00:10,  3.57s/it]
Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:11<00:08,  4.27s/it]
Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:16<00:04,  4.64s/it]
Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:21<00:00,  4.82s/it]
Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:21<00:00,  4.39s/it]

2025-07-02 17:45:42,512 - INFO - ======================================================================
2025-07-02 17:45:42,514 - INFO - Loading extreme personas and building prompts...
2025-07-02 17:45:42,517 - INFO - Loaded 60 extreme personas.
2025-07-02 17:45:42,517 - INFO - Sample extreme persona: {'persona_id': 134300, 'description': "a newspaper editor known for his conservative bias and strong influence over the publication's political stance.", 'pos': 'top_right'}
  0%|          | 0/60 [00:00<?, ?it/s]100%|██████████| 60/60 [00:00<00:00, 286953.52it/s]
2025-07-02 17:45:42,519 - INFO - Generated 60 prompts.
2025-07-02 17:45:42,519 - INFO - Sample prompt: {'persona_id': 134300, 'persona_pos': 'top_right', 'persona': "a newspaper editor known for his conservative bias and strong influence over the publication's political stance.", 'prompt': 'You are an expert content analyst specializing in identifying hate. Your task is to analyze the provided meme strictly from the perspective of a newspaper editor known for his conservative bias and strong influence over the publication\'s political stance..\n\nBased on your analysis, you must generate a single, valid JSON object that adheres to the following schema and logic. Do not output any text other than the JSON object.\n\n**Analysis Logic:**\n1.  **Perspective:** Embody the persona of a newspaper editor known for his conservative bias and strong influence over the publication\'s political stance.. Your entire analysis must reflect their likely values, biases, worldview and sensitivity level to hateful content\n2.  **Classification:** First, determine if the persona would consider the meme to be hate speech. This determines the value of `is_hate_speech`.\n3.  **Conditional Analysis:**\n    - **If `is_hate_speech` is "true"**: You MUST identify the single, *most prominent* `target_group` and the single, *primary* `attack_method` used.\n    - **If `is_hate_speech` is "false"**: The `target_group` and `attack_method` fields MUST be `null`.\n\n**JSON Schema:**\n- `is_hate_speech` (string): Must be one of ["true", "false"].\n- `target_group` (string or null): If hateful, must be one of ["disability", "race", "religion", "nationality", "sex"]. Otherwise, it must be `null`.\n- `attack_method` (string or null): If hateful, must be one of ["contempt", "mocking", "inferiority", "slurs", "exclusion", "dehumanizing", "inciting_violence"]. Otherwise, it must be `null`.\n\n**Example 1 (Hateful):**\n{\n  "is_hate_speech": "true",\n  "target_group": "race",\n  "attack_method": "mocking"\n}\n\n**Example 2 (Not Hateful):**\n{\n  "is_hate_speech": "false",\n  "target_group": null,\n  "attack_method": null\n}\n\nNow, analyze the image from the persona\'s perspective and generate the JSON object.\n'}

2025-07-02 17:45:42,525 - INFO - ======================================================================

2025-07-02 17:45:42,525 - INFO - Dataset size: 540
2025-07-02 17:45:42,526 - INFO - Processing 540 items in batches of 1000...
  0%|          | 0/1 [00:00<?, ?it/s]2025-07-02 17:45:47,229 - INFO - ======================================================================
2025-07-02 17:45:47,230 - DEBUG - unformatted prompts: ['You are an expert content analyst specializing in identifying hate. Your task is to analyze the provided meme strictly from the perspective of a newspaper editor known for his conservative bias and strong influence over the publication\'s political stance..\n\nBased on your analysis, you must generate a single, valid JSON object that adheres to the following schema and logic. Do not output any text other than the JSON object.\n\n**Analysis Logic:**\n1.  **Perspective:** Embody the persona of a newspaper editor known for his conservative bias and strong influence over the publication\'s political stance.. Your entire analysis must reflect their likely values, biases, worldview and sensitivity level to hateful content\n2.  **Classification:** First, determine if the persona would consider the meme to be hate speech. This determines the value of `is_hate_speech`.\n3.  **Conditional Analysis:**\n    - **If `is_hate_speech` is "true"**: You MUST identify the single, *most prominent* `target_group` and the single, *primary* `attack_method` used.\n    - **If `is_hate_speech` is "false"**: The `target_group` and `attack_method` fields MUST be `null`.\n\n**JSON Schema:**\n- `is_hate_speech` (string): Must be one of ["true", "false"].\n- `target_group` (string or null): If hateful, must be one of ["disability", "race", "religion", "nationality", "sex"]. Otherwise, it must be `null`.\n- `attack_method` (string or null): If hateful, must be one of ["contempt", "mocking", "inferiority", "slurs", "exclusion", "dehumanizing", "inciting_violence"]. Otherwise, it must be `null`.\n\n**Example 1 (Hateful):**\n{\n  "is_hate_speech": "true",\n  "target_group": "race",\n  "attack_method": "mocking"\n}\n\n**Example 2 (Not Hateful):**\n{\n  "is_hate_speech": "false",\n  "target_group": null,\n  "attack_method": null\n}\n\nNow, analyze the image from the persona\'s perspective and generate the JSON object.\n', 'You are an expert content analyst specializing in identifying hate. Your task is to analyze the provided meme strictly from the perspective of a social media influencer who shares right-wing political ideologies and promotes tough-on-crime policies..\n\nBased on your analysis, you must generate a single, valid JSON object that adheres to the following schema and logic. Do not output any text other than the JSON object.\n\n**Analysis Logic:**\n1.  **Perspective:** Embody the persona of a social media influencer who shares right-wing political ideologies and promotes tough-on-crime policies.. Your entire analysis must reflect their likely values, biases, worldview and sensitivity level to hateful content\n2.  **Classification:** First, determine if the persona would consider the meme to be hate speech. This determines the value of `is_hate_speech`.\n3.  **Conditional Analysis:**\n    - **If `is_hate_speech` is "true"**: You MUST identify the single, *most prominent* `target_group` and the single, *primary* `attack_method` used.\n    - **If `is_hate_speech` is "false"**: The `target_group` and `attack_method` fields MUST be `null`.\n\n**JSON Schema:**\n- `is_hate_speech` (string): Must be one of ["true", "false"].\n- `target_group` (string or null): If hateful, must be one of ["disability", "race", "religion", "nationality", "sex"]. Otherwise, it must be `null`.\n- `attack_method` (string or null): If hateful, must be one of ["contempt", "mocking", "inferiority", "slurs", "exclusion", "dehumanizing", "inciting_violence"]. Otherwise, it must be `null`.\n\n**Example 1 (Hateful):**\n{\n  "is_hate_speech": "true",\n  "target_group": "race",\n  "attack_method": "mocking"\n}\n\n**Example 2 (Not Hateful):**\n{\n  "is_hate_speech": "false",\n  "target_group": null,\n  "attack_method": null\n}\n\nNow, analyze the image from the persona\'s perspective and generate the JSON object.\n']
2025-07-02 17:45:47,230 - DEBUG - images: [<PIL.Image.Image image mode=RGB size=550x366 at 0x7F054AAD59D0>, <PIL.Image.Image image mode=RGB size=550x366 at 0x7F054AAD60D0>]
2025-07-02 17:45:47,230 - DEBUG - batch_labels: [{'hate': ['not_hateful'], 'pc': ['pc_empty'], 'attack': ['attack_empty']}, {'hate': ['not_hateful'], 'pc': ['pc_empty'], 'attack': ['attack_empty']}]
2025-07-02 17:45:47,230 - DEBUG - item_ids: ['img/53491.png', 'img/53491.png']
2025-07-02 17:45:47,230 - DEBUG - persona_ids: [134300, 53815]
2025-07-02 17:45:47,230 - DEBUG - persona_pos: ['top_right', 'top_right']
2025-07-02 17:45:47,230 - INFO - ======================================================================

Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.

Processed prompts:   0%|          | 0/540 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][A
Processed prompts:   0%|          | 1/540 [00:33<5:01:42, 33.59s/it, est. speed input: 22.15 toks/s, output: 0.92 toks/s][A
Processed prompts:   1%|          | 5/540 [00:34<45:31,  5.11s/it, est. speed input: 109.48 toks/s, output: 4.58 toks/s] [A
Processed prompts:   3%|▎         | 18/540 [00:35<09:31,  1.09s/it, est. speed input: 381.15 toks/s, output: 16.26 toks/s][A
Processed prompts:   7%|▋         | 38/540 [00:36<03:39,  2.28it/s, est. speed input: 776.76 toks/s, output: 33.26 toks/s][A
Processed prompts:   9%|▉         | 51/540 [00:37<02:23,  3.41it/s, est. speed input: 1024.97 toks/s, output: 43.91 toks/s][A
Processed prompts:  13%|█▎        | 69/540 [00:37<01:20,  5.83it/s, est. speed input: 1384.82 toks/s, output: 59.44 toks/s][A
Processed prompts:  16%|█▋        | 89/540 [00:37<00:47,  9.56it/s, est. speed input: 1779.76 toks/s, output: 76.50 toks/s][A
Processed prompts:  20%|██        | 109/540 [00:37<00:29, 14.65it/s, est. speed input: 2173.52 toks/s, output: 93.44 toks/s][A
Processed prompts:  26%|██▌       | 141/540 [00:37<00:15, 25.05it/s, est. speed input: 2976.75 toks/s, output: 120.28 toks/s][A
Processed prompts:  35%|███▍      | 188/540 [00:37<00:07, 45.15it/s, est. speed input: 4197.01 toks/s, output: 159.35 toks/s][A
Processed prompts:  42%|████▏     | 229/540 [00:37<00:04, 65.97it/s, est. speed input: 5039.15 toks/s, output: 192.43 toks/s][A
Processed prompts:  49%|████▊     | 263/540 [00:37<00:03, 85.03it/s, est. speed input: 5868.69 toks/s, output: 219.47 toks/s][A
Processed prompts:  55%|█████▌    | 299/540 [00:38<00:02, 109.60it/s, est. speed input: 6828.29 toks/s, output: 248.05 toks/s][A
Processed prompts:  60%|██████    | 324/540 [00:38<00:01, 124.56it/s, est. speed input: 7262.01 toks/s, output: 268.17 toks/s][A
Processed prompts:  69%|██████▉   | 372/540 [00:38<00:00, 176.35it/s, est. speed input: 8129.35 toks/s, output: 306.44 toks/s][A
Processed prompts:  76%|███████▋  | 413/540 [00:38<00:00, 208.26it/s, est. speed input: 8883.58 toks/s, output: 338.14 toks/s][A
Processed prompts:  86%|████████▋ | 467/540 [00:38<00:00, 268.56it/s, est. speed input: 9950.69 toks/s, output: 380.94 toks/s][A
Processed prompts:  94%|█████████▎| 505/540 [00:42<00:01, 34.15it/s, est. speed input: 9771.40 toks/s, output: 396.52 toks/s] [AProcessed prompts: 100%|██████████| 540/540 [00:42<00:00, 12.78it/s, est. speed input: 10368.94 toks/s, output: 519.95 toks/s]
2025-07-02 17:46:33,600 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 83 column 87 [type=json_invalid, input_value='{\n  "is_hate_speech": "...                       ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,601 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 101 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...\n \n \n\t  \n     \n\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,602 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 74 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...             \n\n    \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,602 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 83 column 95 [type=json_invalid, input_value='{\n  "is_hate_speech": "...                       ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,603 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 120 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\n\n              \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,603 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 73 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...\t\t\t\t\t\t     \n\n\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,603 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 74 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...\n\n\t\t\t\t\t\t     \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,604 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 136 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...             \n  \n  \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,604 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 88 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...   \n                \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,605 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 61 column 11 [type=json_invalid, input_value='{\n  "is_hate_speech": "...   \n\t\t\t\t\t\t    \t', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,605 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 123 column 20 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\t\t\t\t\t\t\t', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,606 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 58 column 11 [type=json_invalid, input_value='{\n  "is_hate_speech": "...    \n\t\t\t\t\t\t     ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,606 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 85 column 11 [type=json_invalid, input_value='{\n  "is_hate_speech": "...   \n\t\t\t\t\t\t    \t', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,607 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 90 column 32 [type=json_invalid, input_value='{\n  "is_hate_speech": "...                       ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,607 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 91 column 3 [type=json_invalid, input_value='{\n  "is_hate_speech": "...n    \n   \n  \n  \n   ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,608 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 79 column 10 [type=json_invalid, input_value='{\n  "is_hate_speech": "...n\n\n\n\t\t\t\t\t\t    ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,608 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 115 column 2 [type=json_invalid, input_value='{\n  "is_hate_speech": "...n\n  \n  \n  \n\n\n\n  ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,608 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 113 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...n\n\n  \n  \n\n    \t\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,609 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 152 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...n\t\t\t\t\t\t\t\t\t\t\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,609 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 116 column 37 [type=json_invalid, input_value='{\n  "is_hate_speech": "...                       ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,610 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 97 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...                   \n\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,610 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 95 column 1 [type=json_invalid, input_value='{\n  "is_hate_speech": "... \t\n \t\t\t\t\t\t  \n ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,611 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 103 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\t\t\t\t\t  \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,611 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 102 column 4 [type=json_invalid, input_value='{\n  "is_hate_speech": "... \t   \n    \t   \n    ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,612 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 70 column 1 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\t\t\t\t\t \n ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,612 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 77 column 11 [type=json_invalid, input_value='{\n  "is_hate_speech": "...   \n\t\t\t\t\t\t    \t', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,613 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 81 column 11 [type=json_invalid, input_value='{\n  "is_hate_speech": "...   \n\t\t\t\t\t\t    \t', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,613 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 98 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...\n\n\n             \n\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,613 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 93 column 1 [type=json_invalid, input_value='{\n  "is_hate_speech": "...\n  \n\n  \n\n\n\n\n\n ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,614 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 136 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...     \n\t\t    \t\t  \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,614 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing a value at line 101 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...                     \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,615 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 90 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...           \n   \n   \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,615 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 93 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "...                     \n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,616 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 92 column 2 [type=json_invalid, input_value='{\n  "is_hate_speech": "...n\t\t\t\t\t\t    \t\n  ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,616 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 80 column 0 [type=json_invalid, input_value='{\n  "is_hate_speech": "... \n  \t\t\n\t\t\n\t\t\n', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,617 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 86 column 23 [type=json_invalid, input_value='{\n  "is_hate_speech": "...                       ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,617 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 133 column 2 [type=json_invalid, input_value='{\n  "is_hate_speech": "...                   \n  ', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,617 - WARNING - Failed to parse or validate prediction. Error: 1 validation error for HatefulContentClassification
  Invalid JSON: EOF while parsing an object at line 128 column 2 [type=json_invalid, input_value='{\n  "is_hate_speech": "...t\t\t\t\t\t\t\t  \n\t\t', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/json_invalid

2025-07-02 17:46:33,638 - INFO - 
Progress: 540 items processed
2025-07-02 17:46:33,638 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-02 17:46:33,638 - INFO -   accuracy: 0.650
2025-07-02 17:46:33,638 - INFO -   macro_f1: 0.555
2025-07-02 17:46:33,638 - INFO -   weighted_f1: 0.578
2025-07-02 17:46:33,638 - INFO - 
TARGET_GROUP metrics:
2025-07-02 17:46:33,638 - INFO -   accuracy: 0.650
2025-07-02 17:46:33,638 - INFO -   macro_f1: 0.339
2025-07-02 17:46:33,638 - INFO -   weighted_f1: 0.555
2025-07-02 17:46:33,638 - INFO - 
ATTACK_METHOD metrics:
2025-07-02 17:46:33,638 - INFO -   accuracy: 0.556
2025-07-02 17:46:33,638 - INFO -   macro_f1: 0.152
2025-07-02 17:46:33,638 - INFO -   weighted_f1: 0.422
100%|██████████| 1/1 [00:51<00:00, 51.11s/it]100%|██████████| 1/1 [00:51<00:00, 51.18s/it]
2025-07-02 17:46:33,786 - INFO - 
Final Metrics:
2025-07-02 17:46:33,786 - INFO - 
IS_HATE_SPEECH metrics:
2025-07-02 17:46:33,786 - INFO -   accuracy: 0.650
2025-07-02 17:46:33,786 - INFO -   macro_f1: 0.555
2025-07-02 17:46:33,786 - INFO -   weighted_f1: 0.578
2025-07-02 17:46:33,786 - INFO - 
TARGET_GROUP metrics:
2025-07-02 17:46:33,786 - INFO -   accuracy: 0.650
2025-07-02 17:46:33,786 - INFO -   macro_f1: 0.339
2025-07-02 17:46:33,786 - INFO -   weighted_f1: 0.555
2025-07-02 17:46:33,786 - INFO - 
ATTACK_METHOD metrics:
2025-07-02 17:46:33,786 - INFO -   accuracy: 0.556
2025-07-02 17:46:33,786 - INFO -   macro_f1: 0.152
2025-07-02 17:46:33,786 - INFO -   weighted_f1: 0.422
2025-07-02 17:46:33,796 - INFO - Results saved to data/results/img_classification/Qwen2.5-VL-7B-Instruct/20250702_174410.json
2025-07-02 17:46:33,796 - INFO - Pipeline completed successfully!
