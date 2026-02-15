from matplotlib import pyplot as plt
import pickle
from chatbot.eval import eval,interface


def plot_loss():
    loss_list = pickle.load(open("chatbot/models/loss_list.pkl","rb"))
    plt.figure(figsize=(50, 8))
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()

if __name__ == '__main__':
    # interface()
    from chatbot.seq2seq import Seq2Seq
    model = Seq2Seq()
    for name, param in model.named_parameters():
        print(name,param.size(),"*"*30)

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("总参数数量和：" + str(k))
