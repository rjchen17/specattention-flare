from typing import Union

import spectrans
import torch
import json
import yaml
import argparse

from vocabs import Vocabulary

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from spectrans.models import FNet
from spectrans.config.models import FNetModelConfig
from pathlib import Path
import wandb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--languages",
                        type=str,
                        default="all",
                        help="Comma separated list of languages to run training on. Use 'all' for all languages. ")
    parser.add_argument("--val_subset",
                        type=str,
                        default="validation-long",
                        help="Which validation set to use. ")

    parser.add_argument("--test_subset",
                        type=str,
                        default="test",
                        help="Which test set to use. ")

    parser.add_argument("--path_to_flare",
                        type=str,
                        default="../flare",
                        help="Path to the 'flare' directory. ")

    parser.add_argument("--path_to_vocabs",
                        type=str,
                        default="../vocabs",
                        help="Path to the 'vocabs' directory. ")
    args = parser.parse_args()
    return args

class FLAREDataset(Dataset):

    def __init__(self, language: Union[str, Path], subset: str = None, vocab_path=None):
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

class EarlyStopper:
    """
    Uses a criterion (e.g. val loss) to end training early, if necessary
    """

    def __init__(self, patience):

        self.patience = patience
        self.min = None
        self.criterion_history = None
        self.stop = False

    def step(self, criterion):
        # If history is empty, set min to current criterion value and add to list
        # Do the same if criterion is less than the min
        if not self.criterion_history or criterion < self.min:
            self.min = criterion
            self.criterion_history = [criterion]

        else:
            self.criterion_history.append(criterion)
            if len(self.criterion_history) > self.patience:
                self.stop = True
                self.criterion_history = []

def test_scheduler_and_es():
    patience = 5
    optimizer = AdamW([torch.tensor([0])], lr=1)

    es = EarlyStopper(patience=patience)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=patience)

    dummy_losses = [10,11,11,11,11,11,11,11,11,11,11,11,11,11]

    for epoch, dummy_loss in enumerate(dummy_losses):


        scheduler.step(dummy_loss)
        es.step(dummy_loss)
        print(f"Loss: {dummy_loss}, LR: {scheduler.get_last_lr()}")

def training_loop(language_path: Path,
                  vocab_path,
                  val_subset,
                  test_subset,
                  model_config_path="../configs/models/default.yml",
                  hyperparameter_config_path="../configs/hyperparameters/default.yml",
                  checkpoint_path="../checkpoints"):

    wandb.init(project="specattention-flare", name=language_path.name, group="validation-short")

    with open(model_config_path, 'r') as model_config_file:
        model_config_args = yaml.safe_load(model_config_file)

    with open(hyperparameter_config_path, 'r') as hyperparameter_config_file:
        hyperparameters = yaml.safe_load(hyperparameter_config_file)

    train_dataset = FLAREDataset(language=language_path, vocab_path=vocab_path)
    val_dataset = FLAREDataset(language=language_path, subset=val_subset, vocab_path=vocab_path)
    test_dataset = FLAREDataset(language=language_path, subset=test_subset, vocab_path=vocab_path)

    # Set model embedding vocab size
    model_config_args["vocab_size"] = len(train_dataset.vocab) + 1
    model_config = FNetModelConfig(**model_config_args)
    model = FNet.from_config(model_config)
    model.embedding.padding_idx = 0  # TODO: not sure if this is necessary

    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], collate_fn=pad_batch)
    val_loader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], collate_fn=pad_batch)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], collate_fn=pad_batch)

    optimizer = AdamW(model.parameters(), lr=hyperparameters["lr"])
    criterion = nn.BCEWithLogitsLoss()

    # Cut lr in half if criterion (e.g. val loss) doesn't decrease after 5 epochs.
    # Parameters taken from FLARE paper.
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=5)

    epochs = hyperparameters["max_train_epochs"]
    model.train()
    # Exit training if criterion (e.g. val loss) doesn't decrease after 10 epochs, also taken from
    # FLARE paper.
    es = EarlyStopper(patience=10)

    # Keep track of best loss for checkpoint saving
    best_loss = float('inf')
    best_checkpoint = None

    for epoch in range(epochs):
        for batch in train_loader:

            output = model(batch["input"])
            output = output.squeeze(1)
            loss = criterion(output, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})

        # Run validation
        val_output = evaluate(model=model, dataloader=val_loader, criterion=criterion)
        val_loss = val_output["loss"]
        val_acc = val_output["accuracy"]

        wandb.log({"val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_loss:
            best_checkpoint = {"epoch": epoch,
                               "model_state": model.state_dict(),
                               "val_acc": val_acc,
                               "val_loss": val_loss,
                                }
            best_loss = val_loss

        # Update lr
        scheduler.step(val_loss)
        es.step(val_loss)
        wandb.log({"lr": scheduler.get_last_lr()})

        if es.stop:
            print(f"Early stop at epoch {epoch}. ")
            break

    model.load_state_dict(best_checkpoint["model_state"])
    test_output = evaluate(model=model, dataloader=test_loader, criterion=criterion)
    test_loss = test_output["loss"]
    test_acc = test_output["accuracy"]

    best_checkpoint["test_loss"] = test_loss
    best_checkpoint["test_acc"] = test_acc

    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    wandb.finish()

    torch.save(best_checkpoint, Path(checkpoint_path) / f"{Path(language_path).name}_checkpoint.pt")
def main():

    args = parse_args()

    if args.languages == "all":
        for language in Path(args.path_to_flare).iterdir():

            if not language.is_dir() or language.name.startswith("."):
                continue

            training_loop(language_path=language,
                          vocab_path=args.path_to_vocabs,
                          val_subset=args.val_subset,
                          test_subset=args.test_subset)

if __name__ == "__main__":
    main()



