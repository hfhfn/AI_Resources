import torch
import torch.nn as nn


#y = wx + b
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        #自定义的代码
        # self.w = torch.rand([],requires_grad=True)
        # self.b = torch.tensor(0,dtype=torch.float,requires_grad=True)
        self.lr = nn.Linear(1,10) #x 中有10列 [5,10]  ----操作[10,2] + b [2]- -->数据只有2列 ,[5,2]
        # self.lr2 = nn.Linear(10,20)
        # self.Lr3 = nn.Linear(20,1)

    def forward(self, x): #完成一次向前的计算
        # y_predict = x*self.w + self.b
        # return y_predict
        out1 = self.lr(x)
        # out2 = self.lr2(out1)
        # out = self.lr3(out2)
        return out1

## 调用模型
if __name__ == '__main__':
    model = MyModel()
    # print(model.parameters())
    for i in model.parameters():
        print(i)
        print("*"*100)
    # y_predict = model([10])  #__call__  --->forward()
    # y_predict = MyModel()(torch.FloatTensor([10]))
    # print(y_predict)
