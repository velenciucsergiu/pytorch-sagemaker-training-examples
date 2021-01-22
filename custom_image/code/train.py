#!/usr/bin/env python

import json
import os
import signal
import sys
import time
import numpy as np
from itertools import chain
from sklearn.metrics import accuracy_score
import pandas as pd
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import data_parser
import boto3
from urllib.parse import urlparse
import shutil


input_dir = '/opt/ml/input'
model_dir = '/opt/ml/model'
output_dir = '/opt/ml/output'
tensorboard_dir = '/opt/tensorboard'
checkpoints = '/opt/ml/checkpoints'
val_dir = "/opt/ml/validation"
tmp_dir = "/opt/ml/tmp"
os.mkdir(tmp_dir)
failure_file = output_dir + '/failure'
data_dir = input_dir + '/data'

writer = SummaryWriter(tensorboard_dir)

terminated = False
global_step = 0
epoch_step = 0
epoch_loss = 0

class DataRenderer:
    def __init__(self, data_dir, channel, epoch, size):
        self.data = bytearray(size)
        self.data_dir = data_dir
        self.channel = channel
        self.epoch = epoch
        self.bytes_read = 1

        self.fifo_path = '{0}/{1}_{2}'.format(self.data_dir, self.channel, self.epoch)
        wait_till_fifo_exists(self.fifo_path)
        self.fifo = open(self.fifo_path, 'rb')
        print("fifo opened")

    def render(self):
        print(self.bytes_read)
        if self.bytes_read > 0 and not terminated:
            self.bytes_read = self.fifo.readinto(self.data)
        else:
            self.bytes_read = 0
            self.data = "Terminated"
        return self.data

    def get_data(self):
        return self.render()


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


def get_validation_data(paths):
    s3 = boto3.client('s3')
    val_dir = "/opt/ml/validation"
    val_files = []
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    for key, value in json.loads(paths).items():
        parsed = urlparse(value)
        print(parsed)
        head, tail = os.path.split(parsed.path)
        print(parsed.netloc)
        print(parsed.path)
        try:
            s3.download_file(Bucket=parsed.netloc, Key=parsed.path[1:], Filename =os.path.join(val_dir, tail))
            val_files.append(os.path.join(val_dir, tail))
        except Exception as e:
            print(e)
    print("Files downloaded")
    return val_files


def save_checkpoints(model, tokenizer, checkpoint_dir, tmp_dir, step):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    current_content = os.listdir(checkpoint_dir)
    for f in current_content:
        #add removing only zip files
        if os.path.splitext(f) == ".zip":
            os.remove(os.path.join(checkpoint_dir, f))
            print(f'File {f} removed')
    directory = os.path.join(tmp_dir,f'checkpoint_step_{step}')
    if not os.path.exists(directory):
        os.mkdir(directory)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    zipfile_name = f'checkpoint_step_{step}'
    shutil.make_archive(os.path.join(tmp_dir, zipfile_name), 'zip', directory)
    print(f'Checkpoint {zipfile_name} created')
    shutil.move(os.path.join(tmp_dir, zipfile_name + ".zip"), os.path.join(checkpoint_dir, zipfile_name + ".zip"))
    print(f'Checkpoint {zipfile_name} saved to {checkpoint_dir}')
    shutil.rmtree(directory)
    return True


def train(model, device, scaler, optimizer, scheduler,
          data, tokenizer, max_len, batch_size, epoch, tb_writer, val_files):
    train_loader = create_dataloader(data, tokenizer, max_len, batch_size)
    model.train()
    batch_loss = 0
    batch_step = 0
    for bid, batch in enumerate(train_loader):
        start = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        model.zero_grad()
        with autocast():
            out = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)
        loss = out[0]
        batch_loss += loss.item()
        global epoch_loss
        epoch_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        torch.cuda.empty_cache()

        global epoch_step
        epoch_step += 1
        global global_step
        global_step += 1
        batch_step += 1

        if (global_step % 100) == 0:
            tb_writer.add_scalar('Loss/train', batch_loss / batch_step, global_step)
            print("Epoch {0} Step {1}/{2} Iteration {3} Loss {4}".format(epoch, bid, len(train_loader),
                                                                         epoch_step, batch_loss / batch_step))
            est = (time.time() - start) * (len(train_loader) - bid)
            print("Time/batch {0}  is {1}".format(bid, time.time() - start))
            print("Estimated time left to end of sequence {}".format(est))
            hyperparameters = json.loads(os.environ["SM_HPS"])
            val_datasets = hyperparameters["s3paths"]
            acc = []
            for val_file in val_files:
                accuracy = validation(model, device, val_file, tokenizer, max_len, batch_size)
                acc.append(accuracy)
            avg_acc = sum(acc)/len(acc)
            tb_writer.add_scalar('Accuracy/test', avg_acc, global_step)
            tb_writer.close()
            save_checkpoints(model = model, tokenizer=tokenizer, checkpoint_dir=checkpoints,
                             tmp_dir=tmp_dir, step=global_step)
            model.train()


def validation(model, device, val_file, tokenizer, max_len, batch_size):
    data = pd.read_csv(val_file, header=None, names=["text", "label"])
    model.eval()
    y_true = []
    y_pred = []
    validation_loader = create_dataloader(data, tokenizer, max_len, batch_size)
    for bid, batch in enumerate(validation_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y_true.append(batch["labels"].numpy().flatten().tolist())

        with torch.no_grad():
            output = model(input_ids, attention_mask)
        prob = nnf.softmax(output[0], dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        y_pred.append(top_class.cpu().numpy().flatten().tolist())
    y_pred = list(chain.from_iterable(y_pred))
    y_true = list(chain.from_iterable(y_true))
    acc = accuracy_score(y_true, y_pred)
    model.train()
    return acc


def main():
    # getting channels, other hyperparameters
    training_channel = 'training'
    trap_signal()
    hyperparameters = json.loads(os.environ["SM_HPS"])
    print(hyperparameters)
    val_files = get_validation_data(hyperparameters["s3paths"])
    num_epochs = int(hyperparameters["epochs"])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=2,
                                                          output_hidden_states=False,
                                                          output_attentions=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(device)

    # setting hyperparameters values
    lr = float(hyperparameters["learning_rate"])
    max_len = int(hyperparameters["max_len"])
    eps = float(hyperparameters["eps"])
    batch_size = int(hyperparameters["batch_size"])
    steps_epoch = int(hyperparameters["steps_epoch"])

    scaler = GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=steps_epoch*num_epochs)
    try:
        for epoch in range(num_epochs):
            global epoch_step
            epoch_step = 0
            global epoch_loss
            epoch_loss = 0
            check_termination()
            render = DataRenderer(data_dir=data_dir, channel="training",
                                     epoch=epoch, size=5000000)
            data = False
            lines = 0
            while data != "Terminated":
                data = render.get_data()
                if data != "Terminated":
                    dataset = data_parser(data)
                    print(len(dataset))
                    lines += len(dataset)
                    train(model=model, device=device, scaler=scaler, optimizer=optimizer,scheduler=scheduler,
                          data=dataset, tokenizer=tokenizer , max_len=max_len, batch_size=batch_size, epoch=epoch,
                          tb_writer=writer, val_files=val_files)
            print("Epoch loss : {}".format(epoch_loss/epoch_step))


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


def create_dataloader(data, tokenizer, max_lenght, batch_size):
    ds = MyDataLoader(data["text"].values, data["label"].values, tokenizer, max_lenght)
    return DataLoader(ds, batch_size=batch_size)


if __name__ == '__main__':
    # As per the SageMaker container spec, the algo takes a 'train' parameter.
    # We will simply ignore this in this dummy implementation.
    main()