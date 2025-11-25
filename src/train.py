from typing import Union

import spectrans
import torch
import json
import yaml
from torch.export import FlatArgsAdapter

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

def evaluate(model, dataloader: DataLoader, criterion):
    with torch.no_grad():
        total = 0
        correct = 0
        loss = 0.0

        model.eval()
        for batch in dataloader:

            output = model(batch["input"]).squeeze(1)
            activation = nn.Sigmoid()
            logits = activation(output)
            labels = batch["label"]

            total += labels.size(0)
            correct += torch.eq(torch.round(logits), labels).sum().item()
            loss += criterion(output, labels).item() * labels.size(0)

    # Take mean
    accuracy = 100 * correct / total
    loss /= len(dataloader.dataset)

    return {"loss": loss, "accuracy": accuracy}

def training_loop(langauge_path,
                  val_subset,
                  test_subset,
                  model_config_path="../configs/models/default.yml",
                  hyperparameter_config_path="../configs/hyperparameters/default.yml"):

    with open(model_config_path, 'r') as model_config_file:
        model_config_args = yaml.load(model_config_file, Loader=yaml.BaseLoader)

    with open(hyperparameter_config_path, 'r') as hyperparameter_config_file:
        hyperparameters = yaml.load(hyperparameter_config_file, Loader=yaml.BaseLoader)

    model_config = FNetModelConfig(**model_config_args)
    model = FNet.from_config(model_config)
    model.embedding.padding_idx = 0 # TODO: not sure if this is necessary

    train_dataset = FLAREDataset(language=langauge_path)
    val_dataset = FLAREDataset(language=langauge_path, subset=val_subset)
    test_dataset = FLAREDataset(language=langauge_path, subset=test_subset)

    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], collate_fn=pad_batch)
    val_loader = DataLoader(val_dataset, hyperparameters["batch_size"], collate_fn=pad_batch)
    test_loader = DataLoader(test_dataset, hyperparameters["batch_size"], collate_fn=pad_batch)

    optimizer = AdamW(model.parameters(), lr=hyperparameters["lr"])
    criterion = nn.BCEWithLogitsLoss()

    # Cut lr in half if criterion (e.g. val loss) doesn't decrease after 5 epochs.
    # Parameters taken from FLARE paper.
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=5)

    epochs = hyperparameters["max_train_epochs"]
    model.train()

    for epoch in range(epochs):
        for batch in train_loader:

            output = model(batch["input"])
            output = output.squeeze(1)
            loss = criterion(output, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Run validation
    val_output = evaluate(model=model, dataloader=val_loader, criterion=criterion)
    val_loss = val_output["loss"]
    val_acc = val_output["acc"]

    # Update lr
    scheduler.step(val_loss)

    model.eval()
    num_correct = 0
    num_total = 0
    for batch in test_loader:
        with torch.no_grad():

            output = model(batch["input"]).squeeze(1)
            activation = nn.Sigmoid()
            logits = activation(output)
            labels = batch["label"]
            num_total += labels.size()[0]
            num_correct += torch.eq(torch.round(logits), labels).sum().item()


def main():
    training_loop(langauge_path="../flare/binary-addition", val_subset="validation-long", test_subset="test")

if __name__ == "__main__":

    main()



