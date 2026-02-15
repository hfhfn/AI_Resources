"""
准备数据集
"""
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision
import conf

def mnist_dataset(train): #准备minist的dataset
    func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )]
    )

    # 1. 准备Mnist数据集
    return  MNIST(root="../data/mnist", train=train, download=False, transform=func)

def get_dataloader(train=True):
    mnist = mnist_dataset(train)
    batch_size = conf.train_batch_size if train else conf.test_batch_size
    return DataLoader(mnist,batch_size=batch_size,shuffle=True)

if __name__ == '__main__':
    for (images,labels) in get_dataloader():
        print(images.size())
        print(labels.size())
        break
