from typing import Union

import spectrans
import torch
import json
import yaml

from vocabs import Vocabulary

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from spectrans.models import FNet
from spectrans.config.models import FNetModelConfig
from pathlib import Path

# Set to path of .json vocab files
vocab_path = Path("../vocabs")

class FLAREDataset(Dataset):

    def __init__(self, language: Union[str, Path], subset: str = None):
        if type(language) is not Path:
            language = Path(language)

        self.vocab = Vocabulary(vocab_path / Path(language.stem).with_suffix(".json"))

        if subset is not None:
            if type(subset) is not Path:
                subset = Path("datasets") / Path(subset)
            language = language / subset

        self.data = open(language / Path("main.tok"), 'r').readlines()
        self.labels = open(language / Path("labels.txt"), 'r').readlines()

        try:
            assert len(self.data) == len(self.labels)
        except AssertionError:
            raise ValueError("Data and labels length mismatch. ")

    def __len__(self):

        return len(self.data)

    def __getitem__(self, item):

        string = self.data[item]
        label = torch.tensor(int(self.labels[item]), dtype=torch.float)
        indexed_string = self.vocab(string.split())
        indexed_string = torch.tensor(indexed_string, dtype=torch.int32)

        return {"input": indexed_string, "label": label}

def pad_batch(batch: list):

    # Pad inputs with zeroes
    input_batch = [sample["input"] for sample in batch]
    max_len = max([len(sample) for sample in input_batch])
    padded_batch = [torch.cat((input_tensor, torch.zeros(max_len - input_tensor.size(0), dtype=torch.int32))) for input_tensor in input_batch]
    padded_batch = torch.stack(padded_batch)

    # Convert label batch from list to tensor
    label_batch = [sample["label"] for sample in batch]
    label_batch = torch.tensor(label_batch)
    return {"input": padded_batch, "label": label_batch}

def training_loop(langauge,
                  eval_subset,
                  model_config_path="../configs/models/default.yml",
                  hyperparameter_config_path="../configs/hyperparameters/default.yml"):

    with open(model_config_path, 'r') as model_config_file:
        model_config_args = yaml.load(model_config_file, Loader=yaml.BaseLoader)

    model_config = FNetModelConfig(**model_config_args)
    model = FNet.from_config(model_config)


    pass

if __name__ == "__main__":

    dyck2_train = FLAREDataset(language="../flare/parity/")
    dyck_2_eval = FLAREDataset(language="../flare/parity/", subset="test")

    train_loader = DataLoader(dyck2_train, batch_size=16, collate_fn=pad_batch)
    eval_loader = DataLoader(dyck_2_eval, batch_size=16, collate_fn=pad_batch)

    embedding_dim = 40

    model = FNet(
        vocab_size=len(dyck2_train.vocab) + 1,
        hidden_dim=embedding_dim,
        num_layers=5,
        max_sequence_length=500,
        num_classes=1
    )
    model.embedding.padding_idx = 0

    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
    print(total)
    #dyck_embedding = nn.Embedding(num_embeddings=len(dyck2_train.vocab) + 1, embedding_dim=embedding_dim, padding_idx=0)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 50

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:

            #output = model(inputs_embeds=dyck_embedding(batch["input"]))
            output = model(batch["input"])
            output = output.squeeze(1)
            loss = criterion(output, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(loss.item())
        print(f"Lr: {scheduler.get_last_lr()}")
        print(f"Loss on epoch {epoch} is {loss.item()}")

    model.eval()
    num_correct = 0
    num_total = 0
    for batch in eval_loader:
        with torch.no_grad():

            #output = model(inputs_embeds=dyck_embedding(batch["input"])).squeeze(1)
            output = model(batch["input"]).squeeze(1)
            activation = nn.Sigmoid()
            logits = activation(output)
            labels = batch["label"]
            num_total += labels.size()[0]
            num_correct += torch.eq(torch.round(logits), labels).sum().item()

    print(num_correct, num_total)
    print(num_correct / num_total)



