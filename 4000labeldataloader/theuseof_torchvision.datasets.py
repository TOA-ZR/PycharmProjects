import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms

cifarSet = torchvision.datasets.CIFAR10(root="../data/cifar",train=True, download=True)
print(cifarSet[0])
img, label = cifarSet[0]
print(img)
print(label)
print(img.format, img.size, img.mode) #Image模块是PIL中最重要的模块，它有一个类叫做image，与模块名称相同
img.show()

mytransform = transforms.Compose([transforms.ToTensor()])

# torch.utils.data.DataLoader
cifarSet = torchvision.datasets.CIFAR10(root="../data/cifar/", train=True, download=True, transform=mytransform)
cifarLoader = torch.utils.data.DataLoader(cifarSet, batch_size=10, shuffle=False, num_workers=2)
#test
for i, data in enumerate(cifarLoader, 0):
    print(data[0][0])
    # PIL
    img = transforms.ToPILImage()(data[0][0])
    img.show()


