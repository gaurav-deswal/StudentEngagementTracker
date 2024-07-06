import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import pickle
import numpy as np

from ResTCN import ResTCN
from utils import get_dataloader

os.chdir(r"F:\RESTCN_CODE")

torch.manual_seed(0)
num_epochs = 25
batch_size = 6
lr = .001
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)


# dataloader = get_dataloader(batch_size,
#                             'train.csv',
#                             os.path.join(os.getcwd(), 'images_train'),
#                             'test.csv',
#                             os.path.join(os.getcwd(), 'images_test'))
dataloader = get_dataloader(batch_size,
                            'csv\\train.csv',
                            os.getcwd(),
                            'csv\\test.csv',
                            os.getcwd())

dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'test']}
print(dataset_sizes, flush=True) # OUTPUT: {'train': 5482, 'test': 1784}



import matplotlib.pyplot as plt
import numpy as np

# Initialize dictionaries to store class counts for train and validation (not test) phases
train_class_counts = {}
val_class_counts = {}

# Iterate through train and validation (not test) phases
from tqdm import tqdm
for phase in ['train']:
    # Initialize class counts dictionary for the current phase
    class_counts = {}
    # Iterate through the dataloader for the current phase
    for inputs, labels in tqdm(dataloader[phase]):

        # Count occurrences of each class label
        for label in labels:
            if label.item() not in class_counts:
                class_counts[label.item()] = 0
            class_counts[label.item()] += 1
    # Store class counts for the current phase
    if phase == 'train':
        train_class_counts = class_counts
    else:
        val_class_counts = class_counts

# Get unique class labels
unique_labels = sorted(set(list(train_class_counts.keys()) + list(val_class_counts.keys())))

# Plotting
# fig, ax = plt.subplots(1,1, figsize=(10, 8))

# Plot train data distribution
plt.bar(unique_labels, [train_class_counts.get(label, 0) for label in unique_labels], color='b')
plt.title('Training Data Distribution')  # corrected from plt.set_title
plt.xlabel('Class')  # corrected from plt.set_xlabel
plt.ylabel('Count')  # corrected from plt.set_ylabel
plt.xticks(unique_labels, ['Not Engaged', 'Barely Engaged', 'Engaged', 'Highly Engaged'])  # corrected from plt.set_xticks and plt.set_xticklabels
plt.show()