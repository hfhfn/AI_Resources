from matplotlib import pyplot as plt
import pickle
from chatbot.eval import  eval

def plot_loss():
    loss_list = pickle.load(open("chatbot/models/loss_list.pkl","rb"))
    plt.figure(figsize=(50, 8))
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()

if __name__ == '__main__':
    eval()