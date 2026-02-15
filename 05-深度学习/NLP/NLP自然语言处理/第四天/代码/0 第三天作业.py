import torch

# 1. ׼������ y = x*w1+b1��׼������
x = torch.rand([50, 3])

w1 = torch.tensor([[3], [4], [7]],dtype=torch.float) #[3,1] --->[50,1]   #y = w1x1 + w2x2 + w3x3  +b
b1 = torch.ones([1,1],dtype=torch.float)
# print(torch.matmul(x, w1).size())
y = torch.matmul(x, w1) + b1

w = torch.rand([3, 1], requires_grad=True)
b = torch.rand([1, 1], requires_grad=True)


def loss_fn(y, y_predict):
    loss = (y_predict - y).pow(2).mean()
    for i in [w, b]:
        # ÿ�η��򴫲�ǰ���ݶ���Ϊ0
        if i.grad is not None:
            i.grad.data.zero_()
    # [i.grad.data.zero_() for i in [w,b] if i.grad is not None]
    loss.backward()
    return loss.data


def optimize(learning_rate):
    # print(w.grad.data,w.data,b.data)
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


for i in range(3000):
    # 2. ����Ԥ��ֵ
    y_predict = torch.matmul(x, w) + b

    # 3.������ʧ���Ѳ������ݶ���Ϊ0�����з��򴫲�
    loss = loss_fn(y, y_predict)

    if i % 100 == 0:
        print(i, loss)
    # 4. ���²���w��b
    optimize(0.1)

print("w", w)
print("b", b)
