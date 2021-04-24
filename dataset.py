import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

import os
import json

import config

def find_tokenizer(name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(name.replace('/', '-'))
    except:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.save_pretrained(os.path.join(config.MODEL_DIR, name.replace('/', '-')))
    return tokenizer

class NQDataset(Dataset):

    def __init__(self, data_args, tokenizer):
        super(NQDataset, self).__init__()

        self.data_args = data_args
        self.tokenizer = tokenizer
        
        self.dataset_size = data_args.dataset_size if data_args.dataset_size <= 307_070 else 307_070
        self.trainset_size = int(self.dataset_size * self.data_args.train_ratio)
        self.evalset_size = self.dataset_size - self.trainset_size

        self.dataset_ = open(os.path.join(config.DATA_DIR, data_args.file_name), 'rt')

        self.set_to_train()

    def set_to_train(self):
        print('** Setting into train mode')
        self.eval = False
        self.len = self.trainset_size

    def set_to_eval(self):
        print('** Setting into eval mode')
        self.eval = True
        self.len = self.evalset_size

    # Get the length of the dataset 
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        line = json.loads(self.dataset_.readline())
        
        model_inputs = self.tokenizer.encode_plus(
            text = line['question_text'],
            text_pair = line['document_text'],
            max_length = self.data_args.max_length,
            padding = 'max_length',
            truncation = 'only_second',
            return_tensors = 'pt'
        )

        model_inputs.update({
            'labels': torch.tensor([
                line['annotations'][0]['long_answer']['start_token'],
                line['annotations'][0]['long_answer']['end_token']
            ], dtype=torch.short)
        })

        return model_inputs

def create_dataloader(dataset:Dataset, batch_size:int, shuffle:bool):
    return DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = 0
    )

def turn_into_torch(inputs:dict) -> dict:
    return {
        'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
        'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
        'labels': torch.tensor(inputs['labels'], dtype=torch.short)
    }

def train_eval_split(cache_dir, train_ratio=0.9):
    len_ = len(os.listdir(cache_dir))

    train_split = int(len_ * train_ratio)
    train_file_names = [f'batch-{i}.json' for i in range(train_split-1)] # Always round down
    print(f'** Length of training files is `data_batch_size` * `{len(train_file_names)}`')

    test_file_names = [f'batch-{i-1}.json' for i in range(train_split, len_)]
    print(f'** Length of test files is `data_batch_size` * `{len(test_file_names)}`')

    return train_file_names, test_file_names

def load_files(fnames:list) -> list:
    dataset = []
    for name in fnames:
        with open(os.path.join(CACHE_DIR, name), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                data = turn_into_torch(data)
                dataset.append(data)
    return dataset if len(dataset) != 0 else None

def read_dataset(dataset_class, dataset_size):
    for i in range(dataset_size):
        dataset_class[i]
