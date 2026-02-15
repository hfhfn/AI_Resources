import torch
import torch.nn as nn
from torch import optim
import time

#定义一个device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.lr = nn.Linear(1,1)

    def forward(self, x):  #x [500,1]  ---> y_predict :[500,1]
        return self.lr(x)

#0 准备数据
x = torch.rand([500,1]).to(device)
y_true = 3*x + 0.8
# print(y_true)

#1. 实例化模型
model = MyModel().to(device)
#2. 实例化优化器
optimizer = optim.Adam(model.parameters(),lr=0.1)
#3. 实例化损失函数
loss_fn = nn.MSELoss()

t0 = time.time()
for i in range(500):
    #4. 梯度置为0
    optimizer.zero_grad()
    #5. 调用模型得到预测值
    y_predict = model(x)
    #6. 通过损失函数，计算得到损失
    loss = loss_fn(y_predict,y_true)
    #7. 反向传播，计算梯度
    loss.backward()
    #8. 更新参数
    optimizer.step()

    #打印部分数据
    if i%10 ==0:
        print(i,loss.item())

for param in model.parameters():
    print(param.item())

print("total cost time:",time.time()-t0) #1.3463990688323975