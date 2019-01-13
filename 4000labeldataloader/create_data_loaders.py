
import torchvision.transforms as transforms

import datasets
import os

args = None
dataset_config = datasets.__dict__[args.dataset]()
print(dataset_config)
dir={'datadir': 'data-local/images/cifar/cifar10/by-image','train_subdir':''}


def create_data_loaders(dir):
    traindir = os.path.join(dir["datadir"], dir["train_subdir"])

train_loader = create_data_loaders(dir)


