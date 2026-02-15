import torch
import torch.nn as nn

batch_size = 10
seq_len = 20
num_embeddings = 100
embedding_dim = 30

input = torch.ones([batch_size,seq_len],dtype=torch.long)

#embedding
embedding = nn.Embedding(100,30)
input_embeded = embedding(input) #[batch_size,seq_len,embedding_dim] #[10,20,30]

#LSTM
lstm = nn.LSTM(input_size=embedding_dim,hidden_size=8,num_layers=2,batch_first=True,bidirectional=True)
# gru = nn.GRU(input_size=embedding_dim,hidden_size=8,num_layers=2,batch_first=True,bidirectional=False)

output,(h_n,c_n) = lstm(input_embeded)

print(output.size())  #[batch_Size,seq_len,8]
print(h_n.size())    #[2,batch_size,hidden_size]
print(c_n.size())

#获取最后一个时间步骤的输出：
#最上一层的输出：
# last_output = output[:,-1,:]
#
# #最上层的h_n:
# h_n_last = h_n[-1,:,:]
# print(last_output.eq(h_n_last))


#双向的LSTM中，最上层的正向的最后一个输出
o1 = output[:,-1,:8] #正向的最后一个输出
o2 = output[:,0,8:]  #反向最后一个输出

#双向LSTM中，最上层正向的h_n
h1 = h_n[-2,:,:] #最上层正向
h2 = h_n[-1,:,:] #最上层反向
print(o1.eq(h1))
print(o2.eq(h2))

