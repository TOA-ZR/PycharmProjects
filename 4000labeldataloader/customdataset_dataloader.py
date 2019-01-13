import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


def default_loader(path):
    return Image.open(path).convert('RGB')



class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        fh = open(label)
        c = 0
        imgs = []
        class_names = []
        for line in fh.readlines():
            if c == 0:
                class_names = [n.strip() for n in line.rstrip().split('	')]
            else:
                cls = line.split()
                fn = cls.pop(0)
                if os.path.isfile(os.path.join(root, fn)):
                    imgs.append((fn, tuple([float(v) for v in cls])))
            c = c + 1
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes

mytransform = transforms.Compose([transforms.ToTensor()])
# torch.utils.data.DataLoader
imgLoader = torch.utils.data.DataLoader(
    myImageFloder(root = "./data/cifar_download/cifar/train", label = "./data/labels/cifar10/4000_balanced_labels/00.txt", transform = mytransform ),
         batch_size= 2, shuffle= False, num_workers= 2)
