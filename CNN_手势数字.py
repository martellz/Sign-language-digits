# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:41:24 2020

@author: 28610
"""

import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import transforms
from finger_sign_dataset import dataset1
import random
'''
CUDA version, edit at 2020.6.7
'''

EPOCH = 15
BATCH_SIZE = 50
LR=0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #图片为1x64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,   #卷积步数
                padding=2,
            ),
            nn.ReLU(),
            #图片为16x64x64
            nn.MaxPool2d(kernel_size=2),
            #图片为16x32x32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            #图片为32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),
            #图片为32x16x16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,5,1,2),
            #图片为64x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),
            #图片为64x8x8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            #图片为128x8x8
            nn.ReLU(),
            nn.MaxPool2d(2),
            #图片为128x4x4
        )
        self.output = nn.Linear(128*4*4, 10)    

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)   
        x = self.conv3(x)
        
        x = self.conv4(x)   #x:(batch,128,4,4)
        x = x.view(x.size(0),-1)    #x:(batch,64*8*8)
        
        output = self.output(x)
        return output
    
   
    
    
train_data = dataset1()     
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)



random_idx = random.sample([i for i in range(len(train_data))], 200)
print('选取作为测试集的索引为:',random_idx)
test_dataset = train_data[random_idx]   #list of: tuple (tensor shape(1,64,64), int:label)
#!!!cuda!!!

test_x = test_dataset[0].cuda()
test_y = torch.from_numpy(test_dataset[1]).cuda()



cnn = CNN()
cnn = cnn.cuda()
#cnn = torch.load('cnn_finger_signal.pkl')


optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    print(epoch)
    for step,(x, y) in enumerate(train_loader):
        #print(step)
        # x = torch.tensor(x,dtype=torch.float32)
        # y = torch.tensor(y,dtype=torch.float32)        
        # !!!using cuda!!!
        x = x.cuda()
        y = y.cuda()
        
        output = cnn(x)
        loss = loss_func(output, y)
        #print('train loss: %.4f' % loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step%50==0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = (pred_y== test_y).sum().item() / test_y.size(0)
            print('Epoch: ',epoch,' | train loss: %.4f' % loss.data, '| test acc:',accuracy)

random_idx2 = random.sample([i for i in range(len(train_data))], 20)
test_x = torch.from_numpy(train_data.X).type(torch.FloatTensor)[random_idx2].cuda()
test_y = torch.from_numpy(train_data.Y).type(torch.FloatTensor)[random_idx2]


test_output = cnn(test_x)

#!!!cuda here!!!
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze() #tensor不支持将data转换成numpy
pred_y = pred_y.cpu()      #转换成cpu数据

print('prediction num: ', pred_y.numpy())
print('real number:', test_y.numpy().astype(int))

#torch.save(cnn, 'cnn_finger_signal.pkl')