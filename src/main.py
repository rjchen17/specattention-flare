import spectrans
import torch

from spectrans.models import FNet

model = FNet(
    vocab_size=2,
    hidden_dim=44,
    num_layers=4,
    max_sequence_length=512,
    num_classes=2
)

total = 0
for parameter in model.parameters():
    total += parameter.numel()
print(total)

