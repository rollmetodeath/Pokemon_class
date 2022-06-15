from torch.utils.data import Dataset
import glob
import os
import random,csv
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from PIL import Image
import torch
import visdom
import time
class myDatasets(Dataset):
    def __init__(self,root,resize,mode):
        super(myDatasets, self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode
        self.name2label = {}

        list_dir = os.listdir(os.path.join(root))
        for file in sorted(list_dir):
            if  not os.path.isdir(os.path.join(self.root,file)):
                continue
            self.name2label[file] = len(self.name2label.keys())
        print(self.name2label)

        self.images,self.labels = self.load_csv('image.csv')

        if mode=='trian':
            self.images = self.images[0:int(len(self.images)*0.6)]
            self.labels= self.labels[0:int(len(self.labels)*0.6)]
        elif mode =='val':
            self.images = self.images[int(len(self.images) * 0.6):int(0.8*len(self.images))]
            self.labels = self.labels[int(len(self.labels) * 0.6):int(0.8*len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    def load_csv(self,filename):
        if  not os.path.exists(os.path.join(self.root,filename)):
            image = []
            for name in  self.name2label.keys():
                image += glob.glob(os.path.join(self.root,name,'*.png'))
                image += glob.glob(os.path.join(self.root,name,'*.jpg'))
                image += glob.glob(os.path.join(self.root,name,'*.jpeg'))
            print(len(image),image)
            random.shuffle(image)
            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                writer = csv.writer(f)
                # "/root/hy-tmp\pokeman"
                for img  in  image:
                    name = img.split(os.sep)[-2]
                    # name = img.split('/')[-2]
                    label = self.name2label[name]
                    writer.writerow([img,label])
                print('written into csv file ',filename)

        images,labels = [],[]
        with open(os.path.join(self.root,filename),mode='r') as f:
            reader = csv.reader(f)
            for row  in reader:
                img , label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert  len(images) == len(labels)
        return images,labels


    def __len__(self):
        return len(self.images)

    def denormalize(self,x):
        mean = [0.485,0.456,0.406]
        std = [0.229,0.224,0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x*std +mean
        return x

    def __getitem__(self, idx):
        img,label = self.images[idx],self.labels[idx]
        trans = tf.Compose([
            lambda x: Image.open(x).convert('RGB'),
            tf.Resize((int(self.resize*1.),int(self.resize*1.25))),
            tf.RandomRotation(15),
            tf.CenterCrop(self.resize),
            tf.ToTensor(),
            tf.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        img = trans(img)
        label = torch.tensor(label)

        return img,label
def main():
    viz = visdom.Visdom()
    datasets = myDatasets("D:\qq文档\pokemon\pokeman",24,'train')
    data = iter(datasets)
    x,y =next(data)
    print(x.shape,y.shape,y)
    viz.image(datasets.denormalize(x),win='sample_x',opts=dict(title='wugui'))

    loader = DataLoader(datasets,batch_size=32,shuffle=True)
    for x, y in loader:
        viz.images(datasets.denormalize(x),nrow=8,win='batch',opts=dict(title='batch'))
        viz.text(str(y.numpy()),win='label',opts=dict(title='batch_y'))
        time.sleep(10)
if __name__ == '__main__':
    main()