import torch
import torch.nn as nn
from params import par
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )

class DeepVO(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(DeepVO,self).__init__()
        # CNN
        self.rnn = nn.LSTM(
                    input_size=int(30720), 
                    hidden_size=par.rnn_hidden_size, 
                    num_layers=2,
                    dropout=par.rnn_dropout_between, 
                    batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)
        # Comput the shape based on diff image size


    def freeze_cnn(self):
        print("Freezing CNN")
        for par in self.conv1.parameters():
            par.requires_grad = False
        for par in self.conv2.parameters():
            par.requires_grad = False
        for par in self.conv3.parameters():
            par.requires_grad = False
        for par in self.conv3_1.parameters():
            par.requires_grad = False
        for par in self.conv4.parameters():
            par.requires_grad = False
        for par in self.conv4_1.parameters():
            par.requires_grad = False
        for par in self.conv5.parameters():
            par.requires_grad = False
        for par in self.conv5_1.parameters():
            par.requires_grad = False
        for par in self.conv6.parameters():
            par.requires_grad = False


        
if __name__ == "__main__":
    M_deepvo = DeepVO(1, 2)
    print(f"Num Parameters: {sum( p.numel() for p in M_deepvo.parameters() if (p.requires_grad))}")
    M_deepvo.rnn = nn.LSTM(
        input_size=int(30720), 
        hidden_size=500, 
        num_layers=2,
        dropout=par.rnn_dropout_between, 
        batch_first=True
    )
    print(f"Num Parameters: {sum( p.numel() for p in M_deepvo.parameters() if (p.requires_grad))}")
    # M_deepvo.freeze_cnn()
    # print(f"Num Parameters: {sum( p.numel() for p in M_deepvo.parameters() if (p.requires_grad))}")


