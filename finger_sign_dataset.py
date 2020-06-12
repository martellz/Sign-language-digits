# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import random

class dataset1(Dataset):
    def __init__(self, 
                 root='./Sign-language-digits-dataset/',):
        self.X = np.load(root+"X.npy")
        self.Y = np.load(root+"Y.npy")
        
        self.X = self.X.reshape((2062,1,64,64))
        #self.Y = self.Y.reshape((2062,10))
        label_r = [np.argmax(item) for item in self.Y]
        self.Y = np.array(label_r)      #Y.shape = (2062,)
        
        self.size = self.X.shape[0]
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        img=self.X[idx]
        img = torch.from_numpy(img)
        label=self.Y[idx]
        return img, label


if __name__ == '__main__':
    test1 = dataset1()
    
    onTest = test1[1800]
    print(onTest[1])
    #print(onTest[0].type)
    #onTest:tuple (array shape(1,64,64), int:label)
    # random_idx = random.sample([i for i in range(len(test1))], 10)
    # print(random_idx)
    # test_dataset = test1[random_idx]
    plt.imshow((onTest[0].numpy().reshape(64,64)*255))
    
