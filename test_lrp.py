import torch
import torch.nn as nn
import numpy as np
import h5py
from zennit.torchvision import ResNetCanonizer
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from zennit.core import Composite
from zennit.rules import ZPlus

# Your existing CNN1D model definition

# Zennit Canonizer for your model
canonizer = SequentialMergeBatchNorm(CNN1D, merge_bn=True)

# Composite to combine different rules
composite = Composite(ZPlus())

# Instantiate Gradient attribution method
gradient = Gradient(model, canonizer=canonizer, composite=composite)

# Example to get relevance for one batch
inputs, labels = next(iter(val_loader))
inputs = inputs.to(device)

# Get relevance
model.eval()
relevance = gradient(inputs)

# Convert relevance scores to numpy for visualization
relevance_np = relevance.cpu().detach().numpy()

# Visualize the relevance for the first sample in the batch
import matplotlib.pyplot as plt

plt.plot(relevance_np[0][0])  # Plot relevance for the first channel (x-axis)
plt.title("Relevance scores for the first time series sample")
plt.xlabel("Time steps")
plt.ylabel("Relevance")
plt.show()
