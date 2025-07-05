# Mock tokenizer
class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"User: {messages[0]['content']}\nAssistant: "

# Import your dataset class
from src.datasets.yoder_text_dataset import YoderIdentityDataset
from utils.util import load_config

config, prompt_templates = load_config()
task_config = config["text_config"]

prompt_template = prompt_templates[task_config["dataset_name"]]["v2"]["template"]


dataset = YoderIdentityDataset(
    data_path="data/raw/yoder_data/sampled/identity_hate_corpora.jsonl",
    tokenizer=MockTokenizer(),
    max_samples=None,
    seed=22,
    fold="test",
    target_group_size=1,
    extreme_pos_personas_path="data/results/extreme_pos_personas/Qwen2.5-32B-Instruct/extreme_pos_corners_100.pkl",
    prompt_template=prompt_template,
)

# Print first N items to file
N = 5
with open("debug_output.txt", "w") as f:
    for i in range(N):
        try:
            item = dataset[i]
            f.write(f"Item {i}:\n")
            f.write(f"{item}\n")
            f.write("-" * 50 + "\n")
        except Exception as e:
            f.write(f"Error at item {i}: {e}\n")
            f.write("-" * 50 + "\n")

print(f"Debug output written to debug_output.txt")