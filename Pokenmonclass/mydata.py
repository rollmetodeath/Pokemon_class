import csv
import random
import os
import glob

import visdom
from PIL import Image
import time
import torch
from torchvision import transforms as tf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Mydatasets(Dataset):
    def __init__(self,root,size,style):
        self.size = size
        self.style = style
        self.root = root
        self.nameclass = {}
        imagefiles = os.listdir(os.path.join(self.root))
        for file in sorted(imagefiles):
            if not os.path.isdir(os.path.join(self.root,file)):
                continue
            self.nameclass[file] = len(self.nameclass.keys())
        print(self.nameclass)
        self.images,self.labels = self.load_data('load_data.csv')
        if style =='train':
            self.images = self.images[0:int(len(self.images)*0.6)]
            self.labels=self.labels[0:int(len(self.labels)*0.6)]
        elif style =='val':
            self.images = self.images[int(len(self.images)*0.6):int(len(self.images)*0.8)]
            self.labels = self.labels[int(len(self.labels)*0.6):int(len(self.labels)*0.8)]
        else:
            self.images = self.images[int(len(self.images)*0.8):]
            self.labels = self.labels3[int(len(self.labels)*0.8):]
    def load_data(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images =[]
            for file in self.nameclass.keys():
                images += glob.glob(os.path.join(self.root,file,'*.png'))
                images += glob.glob(os.path.join(self.root,file,'*.jpg'))
                images += glob.glob(os.path.join(self.root,file,'*jpeg'))
            random.shuffle(images)
            with open(os.path.join(self.root,filename),'w',newline='') as f:
                writer = csv.writer(f)
                for image in images:
                    name = image.split(os.sep)[-2]
                    label = self.nameclass[name]
                    print(image)
                    writer.writerow([image,label])
                print('写入完毕！')
        images,labels = [],[]
        with open(os.path.join(self.root,filename),'r') as f:
            reader = csv.reader(f)
            for row in reader:
                image,label = row
                label = int(label)
                images.append(image)
                labels.append(label)
        return images,labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image,label = self.images[idx],self.labels[idx]
        trans = tf.Compose([
            lambda x: Image.open(x).convert('RGB'),
            tf.Resize((int(self.size*1.25),int(self.size*1.25))),
            tf.RandomRotation(15),
            tf.CenterCrop(self.size),
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = trans(image)
        return image,torch.tensor(label)
    def denprintimage(self,x):
        m = [0.485, 0.456, 0.406]
        n = [0.229, 0.224, 0.225]
        m = torch.tensor(m).unsqueeze(1).unsqueeze(1)
        n = torch.tensor(n).unsqueeze(1).unsqueeze(1)
        x = x*n+m
        return x

def main():
    # root = "D:\数据集\pokemon\pokeman"
    # train_data = Mydatasets(root,224,'train')
    # # val_data = Mydatasets(root,224,'val')
    # # test_data = Mydatasets(root,224,'test')
    # # train_loader = DataLoader(train_data,shuffle=True,batch_size=32,num_workers=2)
    # # val_loader = DataLoader(val_data,shuffle=True,batch_size=32,num_workers=2)
    # # test_data = DataLoader(test_data,shuffle=True,batch_size=32,num_workers=2)
    # # viz = visdom.Visdom()
    # datasets = iter(train_data)
    # images,labels = next(datasets)
    # print(images.shape)
    # print(labels.shape)

    datasets = Mydatasets("D:\数据集\pokemon\pokeman", 24, 'train')
    data = iter(datasets)
    x, y = next(data)
    print(x.shape, y.shape, y)
if __name__ == '__main__':
    main()
