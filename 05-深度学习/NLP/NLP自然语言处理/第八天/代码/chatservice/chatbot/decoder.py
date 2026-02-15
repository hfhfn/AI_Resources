"""
实现解码器
"""
import torch.nn as nn
import config
import torch
import torch.nn.functional as F
import numpy as np
import random


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.embedding = nn.Embedding(num_embeddings=len(config.target_ws),
                                      embedding_dim=config.chatbot_decoder_embedding_dim,
                                      padding_idx=config.target_ws.PAD)

        #需要的hidden_state形状：[1,batch_size,64]
        self.gru = nn.GRU(input_size=config.chatbot_decoder_embedding_dim,
                          hidden_size=config.chatbot_decoder_hidden_size,
                          num_layers=config.chatbot_decoder_number_layer,
                          bidirectional=False,
                          batch_first=True,
                          dropout=config.chatbot_decoder_dropout)

        #假如encoder的hidden_size=64，num_layer=1 encoder_hidden :[2,batch_sizee,64]

        self.fc = nn.Linear(config.chatbot_decoder_hidden_size,len(config.target_ws))

    def forward(self, encoder_hidden,target):
        # print("target size:",target.size())
        #第一个时间步的输入的hidden_state
        decoder_hidden = encoder_hidden  #[1,batch_size,128*2]
        #第一个时间步的输入的input
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor([[config.target_ws.SOS]]*batch_size).to(config.device)         #[batch_size,1]
        # print("decoder_input:",decoder_input.size())


        #使用全为0的数组保存数据，[batch_size,max_len,vocab_size]
        decoder_outputs = torch.zeros([batch_size,config.chatbot_target_max_len,len(config.target_ws)]).to(config.device)

        if random.random()>0.5:

            for t in range(config.chatbot_target_max_len):
                decoder_output_t,decoder_hidden = self.forward_step(decoder_input,decoder_hidden)
                decoder_outputs[:,t,:] = decoder_output_t


                #获取当前时间步的预测值
                value,index = decoder_output_t.max(dim=-1)
                decoder_input = index.unsqueeze(-1)  #[batch_size,1]
                # print("decoder_input:",decoder_input.size())
        else:
            for t in range(config.chatbot_target_max_len):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs[:, t, :] = decoder_output_t
                #把真实值作为下一步的输入
                decoder_input = target[:,t].unsqueeze(-1)
                # print("decoder_input size:",decoder_input.size())
        return decoder_outputs,decoder_hidden


    def forward_step(self,decoder_input,decoder_hidden):
        '''
        计算一个时间步的结果
        :param decoder_input: [batch_size,1]
        :param decoder_hidden: [1,batch_size,128*2]
        :return:
        '''

        decoder_input_embeded = self.embedding(decoder_input)
        # print("decoder_input_embeded:",decoder_input_embeded.size())

        #out:[batch_size,1,128*2]
        #decoder_hidden :[1,bathc_size,128*2]
        print(decoder_hidden.size())
        out,decoder_hidden = self.gru(decoder_input_embeded,decoder_hidden)
        # print("decoder_hidden size:",decoder_hidden.size())
        #out ：【batch_size,1,hidden_size】

        out_squeezed = out.squeeze(dim=1) #去掉为1的维度
        out_fc = F.log_softmax(self.fc(out_squeezed),dim=-1) #[bathc_size,vocab_size]
        # out_fc.unsqueeze_(dim=1) #[bathc_size,1,vocab_size]
        # print("out_fc:",out_fc.size())
        return out_fc,decoder_hidden

    def evaluate(self,encoder_hidden):

        # 第一个时间步的输入的hidden_state
        decoder_hidden = encoder_hidden  # [1,batch_size,128*2]
        # 第一个时间步的输入的input
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor([[config.target_ws.SOS]] * batch_size).to(config.device)  # [batch_size,1]
        # print("decoder_input:",decoder_input.size())

        # 使用全为0的数组保存数据，[batch_size,max_len,vocab_size]
        decoder_outputs = torch.zeros([batch_size, config.chatbot_target_max_len, len(config.target_ws)]).to(
            config.device)

        predict_result = []
        for t in range(config.chatbot_target_max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t

            # 获取当前时间步的预测值
            value, index = decoder_output_t.max(dim=-1)
            predict_result.append(index.cpu().detach().numpy()) #[[batch],[batch]...]
            decoder_input = index.unsqueeze(-1)  # [batch_size,1]
            # print("decoder_input:",decoder_input.size())
            # predict_result.append(decoder_input)
        #把结果转化为ndarray，每一行是一条预测结果
        predict_result = np.array(predict_result).transpose()
        return decoder_outputs, predict_result

