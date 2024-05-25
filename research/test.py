from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import os

# Define paths (replace with your actual data locations)
data_dir = Path("C:/Users/nikil/Documents/Project/NLP Projects/textsum1/textsum/cnn_dailymail")
train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"
validation_path = data_dir / "validation.csv"

# Load data using Path objects
data_files = {
    "train": train_path,
    "test": test_path,
    "validation": validation_path
}

dataset = load_dataset("csv", data_files=data_files)


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str


class ConfigurationManager:
    def __init__(self, config_filepath="config.yaml", params_filepath="params.yaml"):
        # Implement your YAML reading logic here (using PyYAML or similar)
        # For now, we'll assume default values
        self.config = {
            "data_transformation": {
                "root_dir": "artifacts/data_transformation",
                "data_path": "data/cnn_dailymail",  # Placeholder, update based on your data structure
                "tokenizer_name": "google/pegasus-cnn_dailymail"
            }
        }

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config["data_transformation"]
        root_dir = Path(config["root_dir"])
        data_path = Path(config["data_path"])
        tokenizer_name = config["tokenizer_name"]

        # Create directories if they don't exist (improve error handling if needed)
        os.makedirs(root_dir, exist_ok=True)

        return DataTransformationConfig(root_dir, data_path, tokenizer_name)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['dialogue'], max_length=1024, truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'], max_length=128, truncation=True
            )

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))


if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()
    except Exception as e:
        print(f"An error occurred: {e}")
