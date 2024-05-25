import os
from pathlib import Path
import tqdm as notebook_tqdm
from src.nlp.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from src.nlp.entity import DataTransformationConfig


data_dir = Path("C:/Users/nikil/Documents/Project/NLP Projects/textsum1/artifacts/data_ingestion")
train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"
validation_path = data_dir / "validation.csv"

data_files = {
    "train": str(train_path),
    "test": str(test_path),
    "validation": str(validation_path)
}

dataset = load_dataset("csv", data_files=data_files)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    
    def convert_examples_to_features(self,example_batch):
        input_encodings = self.tokenizer(example_batch['article'] , max_length = 5000, truncation = True )
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['highlights'], max_length = 500, truncation = True )
            
        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    

    def convert(self):
        #dataset_samsum = load_from_disk(self.config.data_path)
        dataset1 = dataset
        dataset1 = dataset1.map(self.convert_examples_to_features, batched = True)
        dataset1.save_to_disk(os.path.join(self.config.root_dir,"dataset"))