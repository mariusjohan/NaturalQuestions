import torch
from torch import nn

import transformers
from transformers import AutoModel
from transformers import AdamW
from transformers.modeling_outputs import QuestionAnsweringModelOutput as QAModelOutputs
from transformers.modeling_outputs import ModelOutput

from typing import Callable, Dict

def find_model(model_name):
    try:
        model = AutoModel.from_pretrained(name.replace('/', '-'))
    except:
        model = AutoModel.from_pretrained(name)
        model.save_pretrained(os.path.join(config.MODEL_DIR, name.replace('/', '-')))
    return model

def create_optimizer(Model:nn.Module, training_args:Dict) -> Callable:
    params = Model.parameters()
    optimizer = AdamW(
        params = params,
        lr = training_args.learning_rate,
        eps = training_args.adam_epsilon,
        weight_decay = training_args.weight_decay,
        correct_bias = True
    )
    return optimizer

def loss_fn(preds:list, labels:list):
    start_positions, end_positions = labels.split(1, dim = -1)
    start_positions = start_positions.squeeze(-1)
    end_positions = end_positions.squeeze(-1)

    ignored_index = preds[0].size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    _loss = nn.CrossEntropyLoss(ignore_index = ignored_index)
    start_loss = _loss(preds[0], start_positions)
    end_loss = _loss(preds[1], end_positions)

    avg_loss = (start_loss + end_loss) / 2
    return avg_loss

class Net(nn.Module):

    def __init__(self, model_args, loss_fn):
        super(Net, self).__init__()

        self.model_args = model_args

        self.model = find_model(model_args.model_name)
        self.config = self.model.config
        self.loss_fn = loss_fn

        self.linear = nn.Linear(self.config.hidden_size, 2)
        self.activation = model_args.activation
        self.dropout = nn.Dropout(model_args.dropout_rate)

    def forward(self, input_ids:list, attention_mask:list, token_type_ids:list = None, labels:list = None) -> QAModelOutputs:
        model_output = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids
        )
        sequence_output = model_output[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)

        # Format the outputs to predict start and end token
        start_logits, end_logits = logits.split(1, dim = -1)
        start_logits = torch.squeeze(start_logits, axis=-1)
        end_logits = torch.squeeze(end_logits, axis=-1)

        preds = [start_logits, end_logits]

        # Calculate loss
        loss = None
        if type(labels) != type(None):
            loss = self.loss_fn(preds = preds, labels = labels)

        return QAModelOutputs(
            loss = loss,
            start_logits = start_logits,
            end_logits = end_logits
        )
