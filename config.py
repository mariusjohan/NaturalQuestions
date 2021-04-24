import os

from torch import nn

import dataclasses
from dataclasses import dataclass, field
from transformers import TrainingArguments

from typing import Callable

DATA_DIR = './data'
CACHE_DIR = './cache'
OUTPUT_DIR = './model'
MODEL_DIR = 'C://Users//mariu//.cache//huggingface'

def create_env():
    global DATA_DIR, CACHE_DIR, MODEL_DIR

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

@dataclass
class ModelArgs:
    model_name:str = field()
    tokenizer_name:str = field(default = None)
    config_name:str = field(default = None)
    cache_dir:str = field(default = None)
    
    # Hyper parameters
    inputs:int = field(default = 768)
    outputs:int = field(default = 2)
    activation:Callable = field(default = nn.Sigmoid())
    dropout_rate:float = field(default = .15)
    learning_rate:float = field(default = 2.5e-4)

@dataclass
class DataArgs:
    file_name:str = field(default = 'train.json')
    max_length:int = field(default = 384) # Both the context and query
    dataset_size:int = field(default = 100_000)
    train_ratio:int = field(default = 0.9)
    save_to_cache:str = field(default = True)

def TrainingArguments_builder(
    output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    evaluation_strategy='epoch',
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    learning_rate=2.5e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    num_train_epochs=1.0,
    warmup_steps=250,
    logging_steps=100,
    save_steps=7500,
    eval_steps=100
):
    return TrainingArguments(
        output_dir = output_dir,
        overwrite_output_dir = overwrite_output_dir,
        do_train = do_train,
        do_eval = do_eval,
        evaluation_strategy = evaluation_strategy ,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        adam_epsilon = adam_epsilon,
        max_grad_norm = max_grad_norm,
        num_train_epochs = num_train_epochs,
        warmup_steps = warmup_steps,
        logging_steps = logging_steps,
        save_steps = save_steps,
        eval_steps = eval_steps
    )