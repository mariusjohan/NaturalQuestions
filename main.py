from tqdm import tqdm

from modeling import *
from dataset import *
from config import *

from time import time

def create_train_args(model_name):
    model_args = config.ModelArgs(model_name)
    data_args = config.DataArgs(max_length = 512, dataset_size=100)
    training_args = config.TrainingArguments_builder(output_dir = config.OUTPUT_DIR, per_device_train_batch_size = 1)

    tokenizer = find_tokenizer(model_name)
    dataset = NQDataset(data_args, tokenizer)
    trainset = create_dataloader(
        dataset = dataset,
        batch_size = training_args.per_device_train_batch_size,
        shuffle = True
    )

    model = Net(model_args, loss_fn)

    optimizer = create_optimizer(model, training_args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 1
    
    return {
        'model': model,
        'optimizer': optimizer,
        'data_loader': trainset,
        'device': device,
        'training_args': training_args,
        'epochs': epochs
    }

def train(model, optimizer, data_loader, device, training_args, epochs):
    # Metrics
    losses = []

    for epoch in range(epochs):
        epoch_start_time = time()

        batch_losses = []

        # Iterate over each batch
        for batch_i, batch in enumerate(data_loader):
            batch_start_time = time()

            input_ids = batch['input_ids']
            input_ids = input_ids.to(device, dtype=torch.long)
            input_ids = torch.squeeze(input_ids, 1)

            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.to(device, dtype=torch.long)
            attention_mask = torch.squeeze(attention_mask, 1)

            if type(batch.get('token_type_ids')) != type(None):
                token_type_ids = batch['token_type_ids']
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                token_type_ids = torch.squeeze(token_type_ids, 1)
            else:
                token_type_ids = batch.get('token_type_ids')

            labels = batch['labels']
            labels = labels.to(device, dtype=torch.long)

            # Zero out the gradients
            # Maybe move this part out of the batch iterator
            optimizer.zero_grad()

            model_outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                labels = labels
            )

            loss = model_outputs.loss
            preds = model_outputs.start_logits, model_outputs.end_logits

            # Backpropogate 
            loss.requires_loss = True
            loss.backward()

            # Optimize the model
            optimizer.step()

            batch_losses.append(loss.cpu().detach().numpy().tolist())
            if batch_i % 50 == 0:
                print(
                    'Batch:', batch_i, ' ',
                    'Time:', f'{time() - batch_start_time + 0.1**4:.2}  ',
                    'Loss:', f'{float(loss):.4} '
                )
            
        # Report the epoch results
        avg_epoch_loss = np.mean(np.array(batch_losses))
        print('-'*5, 'EPOCH RESULTS', '-'*5)
        print(
            'Epoch:', epoch, ' ',
            'Time:', time() - epoch_start_time, ' ',
            'Loss:', avg_epoch_loss, ' ',
        )
        print('-' * len('-'*5 + ' EPOCH RESULTS ' + '-'*5))