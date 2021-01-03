#!/usr/bin/env python
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torch
import itertools
import json
import os
import signal
import sys
import pandas as pd
import time

input_dir = '/opt/ml/input'
model_dir = '/opt/ml/model'
output_dir = '/opt/ml/output'


class MyDataLoader:
    def __init__(self, review, label, tokenizer, max_lenght):
        self.review = review
        self.label = label
        self.tokenizer = tokenizer
        self.max_lenght = max_lenght

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        reviews = str(self.review[idx])
        label = self.label[idx]
        encoding = self.tokenizer.encode_plus(reviews,
                                              add_special_tokens=True,
                                              max_length=self.max_lenght,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              return_token_type_ids=False,
                                              pad_to_max_length=True,
                                              truncation=True
                                              )
        return {
            "review_text": reviews,
            "input_ids": encoding['input_ids'].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def create_dataloader(data, tokenizer, max_lenght, batch_size):
    ds = MyDataLoader(data["text"].values, data["label"].values, tokenizer, max_lenght)
    return DataLoader(ds, batch_size=batch_size)
#hardcoding num_epochs, even if can be supplied from outside as hyperparameter
num_epochs = 1
channel_name = 'training'
terminated = False

#main function to run
def main():
    #loading tokenizers and bert model.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=2,
                                                          output_hidden_states=False,
                                                          output_attentions=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    print(device)

    #setting hyper parameter. Again, not really necessary to hardcode it, can be supplied as hyperparameters
    lr = 2e-5
    MAX_LEN = 128
    eps = 1e-8
    batch_size = 20
    scaler = GradScaler()
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps, correct_bias=False)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=2500*num_epochs)
    # trapping signals and responding to them appropriately is required by
    # SageMaker spec
    trap_signal()

    # writing to a failure file is also part of the spec
    failure_file = output_dir + '/failure'
    data_dir = input_dir + '/data'

    try:
        # we're allocating a byte array here to read data into, a real algo
        # may opt to prefetch the data into a memory buffer and train in
        # in parallel so that both IO and training happen simultaneously
        data = bytearray(5000000)
        total_read = 0
        total_duration = 0
        for epoch in range(num_epochs):
            steps = 0
            epoch_loss = 0
            check_termination()
            epoch_bytes_read = 0
            # As per SageMaker Training spec, the FIFO's path will be based on
            # the channel name and the current epoch:
            fifo_path = '{0}/{1}_{2}'.format(data_dir, channel_name, epoch)
            print(epoch_bytes_read)
            # Usually the fifo will already exist by the time we get here, but
            # to be safe we should wait to confirm:
            wait_till_fifo_exists(fifo_path)
            ##buffering should be not zero, otherwise the file will not be read in sequences
            with open(fifo_path, 'rb', buffering=5000000) as fifo:
                print('opened fifo: %s' % fifo_path)
                start = time.time()
                bytes_read = 1
                while bytes_read > 0 and not terminated:
                    bytes_read = fifo.readinto(data)
                    print(bytes_read)
                    b = data.splitlines()
                    d = []
                    for s in b:
                        try:
                            text = s.decode("utf-8")
                        except Exception as e:
                            text = s.decode("utf-8", "ignore")
                        if '"' in text:
                            try:
                                d.append(list(eval(text)))
                            except Exception as e:
                                pass
                        else:
                            try:
                                d.append(text.split(','))
                            except Exception as e:
                                pass
                    df = [x for x in d if len(x) == 2 and str(x[1]).isdigit()]
                    if len(df) > 0 :
                        df = pd.DataFrame(df, columns=["text", "label"])
                        df = df.astype({"text":str, "label":int})
                        print(len(df))
                    train_loader = create_dataloader(df, tokenizer, MAX_LEN, batch_size)
                    for bid, batch in enumerate(train_loader):
                        steps += 1
                        start = time.time()
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)

                        model.zero_grad()
                        with autocast():
                            out = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)

                        loss = out[0]
                        epoch_loss += loss.item()
                        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        torch.cuda.empty_cache()
                        if (steps % 200) == 0:
                            print("Epoch {0} Step {1}/{2} Iteration {3} Loss {4}".format(epoch, bid, len(train_loader), steps, epoch_loss / steps))
                            est = (time.time() - start) * (len(train_loader) - bid)
                            print("Time/batch {0}  is {1}".format(bid, time.time() - start))
                            print("Estimated time left to end of sequence {}".format(est))
                    total_read += bytes_read
                    epoch_bytes_read += bytes_read

                print("Epoch {0} loss is {1}".format(epoch, epoch_loss / steps))
                duration = time.time() - start
                total_duration += duration
                epoch_throughput = epoch_bytes_read / duration / 1000000
                print('Completed epoch %s; read %s bytes; time: %.2fs, throughput: %.2f MB/s'
                      % (epoch, epoch_bytes_read, duration, epoch_throughput))

        # now write a model, again, totally meaningless contents:
        model.save_pretrained(model_dir)
    except Exception:
        print('Failed to train: %s' % (sys.exc_info()[0]))
        touch(failure_file)
        raise


def check_termination():
    if terminated:
        print('Exiting due to termination request')
        sys.exit(0)


def wait_till_fifo_exists(fname):
    print('Wait till FIFO available: %s' % (fname))
    while not os.path.exists(fname) and not terminated:
        time.sleep(.1)
    check_termination()


def touch(fname):
    open(fname, 'wa').close()


def on_terminate(signum, frame):
    print('caught termination signal, exiting gracefully...')
    global terminated
    terminated = True


def trap_signal():
    signal.signal(signal.SIGTERM, on_terminate)
    signal.signal(signal.SIGINT, on_terminate)


if __name__ == '__main__':
    # As per the SageMaker container spec, the algo takes a 'train' parameter.
    # We will simply ignore this in this dummy implementation.
    main()
