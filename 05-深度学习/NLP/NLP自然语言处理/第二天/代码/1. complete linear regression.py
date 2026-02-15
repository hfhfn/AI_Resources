import torch
import matplotlib.pyplot as plt

learning_rate = 0.1

# 1. 准备数据 #y = 3x + 0.8
x = torch.randn([500,1])
y_true = 3*x + 0.8

# 2. 计算预测值 y_pred = x * w + b
w = torch.rand([],requires_grad=True)
b = torch.tensor(0,dtype=torch.float,requires_grad=True)

for k in range(30):
    for i in [w,b]:
        if i.grad is not None:
            i.grad.data.zero_()

    y_predict = x * w + b
    # 3. 计算损失，把参数的梯度置为0，进行反向传播
    loss =  (y_predict-y_true).pow(2).mean()

    loss.backward()
    # 3.1 能够得到w和b的梯度
    # 4. 更新参数
    w.data = w.data - learning_rate * w.grad
    b.data = b.data - learning_rate * b.grad
    if k%10 == 0:
        print(k,loss.item(),w.item(),b.item())
# print(w,b)

#绘图
plt.figure(figsize=(20,8))
plt.scatter(x.numpy(),y_true.numpy())

y_predict =  x * w + b
plt.plot(x.numpy(),y_predict.detach().numpy(),c="red")
plt.show()
