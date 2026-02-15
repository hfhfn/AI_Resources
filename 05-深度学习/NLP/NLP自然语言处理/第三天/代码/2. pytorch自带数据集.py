from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision

func = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )]
)

#1. 准备Mnist数据集
mnist = MNIST(root="./data/mnist",train=True,download=False,transform=func)
# print(mnist)
# print(mnist[0])
# print(len(mnist))

# print(mnist[0][0].show())

#2. 准备数据加载器
dataloader = DataLoader(mnist,batch_size=64,shuffle=True)



if __name__ == '__main__':

    for idx,(images,labels) in enumerate(dataloader):
        print(idx)
        print(images) #[64,1,28,28]
        print(labels) #[64]
        break