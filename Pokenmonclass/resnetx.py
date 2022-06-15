import torch
from torch import nn
import os
import glob
from torch.utils.data import DataLoader
import torchvision
from torch import optim
from Datasetsload import myDatasets
# from resnet18 import resnet18
import visdom
from torchvision.models import resnet18
viz = visdom.Visdom()
batch_size = 32
lr = 1e-3
epoches = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)


train = myDatasets("hy-tmp/pokeman",224,mode='trian')
val= myDatasets("/hy-tmp/pokeman",224,mode='val')
test = myDatasets("/hy-tmp/pokeman",224,mode='test')
trian_loader = DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=2)
val_loader = DataLoader(val,batch_size=batch_size,num_workers=2)
test_loader = DataLoader(test,batch_size=batch_size,num_workers=2)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1]))
        return x.view(-1,shape)
def evalute(model,loader):
    correct = 0
    total = len(loader.dataset)
    for  x,y  in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct = torch.eq(pred,y).sum().float().item()
    return  correct/total


def main():
    # x = torch.randn(15,3,224,224)
    Train_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(Train_model.children())[:-1],
                          Flatten(),
                          nn.Linear(512,5)
                          ).to(device)
    # model = resnet18().to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc ,best_epoch =0,0
    global_step = 0
    viz.line([0],[-1],win='loss',opts=dict(title='loss'))
    viz.line([0],[-1],win='val_acc',opts = dict(title='val_acc'))

    for epoch in  range(epoches):
        for step,(x,y) in enumerate(trian_loader):
            x,y = x.to(device),y.to(device)
            print(y.shape)
            logits = model(x)
            print(logits.shape)
            loss = criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', opts=dict(title='loss'),update='append')
            global_step+=1
        if epoch%2 == 0:
            val_acc = evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(),'best.mdl')

                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best :acc',best_acc,'best epoch',best_epoch)
    model.load_state_dict(torch.load('best.mdl'))
    test_acc = evalute(model,test_loader)
    print('test acc',test_acc)

if __name__ =='__main__':
    main()

