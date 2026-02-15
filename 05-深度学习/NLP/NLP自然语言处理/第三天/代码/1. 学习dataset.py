import torch
from torch.utils.data import Dataset,DataLoader

class SmsDataset(Dataset):
    def __init__(self):
        self.file_path = "./data/SMSSpamCollection"
        self.lines  = open(self.file_path).readlines()

    def __getitem__(self, index):
        line = self.lines[index].strip()
        label = line.split("\t")[0]
        content = line.split("\t")[1]
        return label,content

    def __len__(self):
        return len(self.lines)


#使用DataLoader
sms_dataset = SmsDataset()
# print(sms_dataset[0])
dataloader = DataLoader(sms_dataset,batch_size=2,shuffle=True,drop_last=True)

if __name__ == '__main__':
    for idx,(labels,contents) in enumerate(dataloader):
        print(idx)
        print(labels)
        print(contents)
        break
    print(len(sms_dataset))
    print(len(dataloader))