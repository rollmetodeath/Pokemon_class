import torch
from torch import nn
class bottle(nn.Module):
    def __init__(self,inputchanel,outchanel,stride=1):
        super(bottle, self).__init__()
        self.conv1 = nn.Conv2d(inputchanel,outchanel,stride=stride,kernel_size=3,padding=0)
        self.batchnorm1 = nn.BatchNorm2d(outchanel)
        self.conv2 = nn.Conv2d(outchanel,outchanel,stride=1,kernel_size=3,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(outchanel)
        self.relu = nn.ReLU(inplace=True)
        self.change = nn.Sequential()
        if inputchanel!= outchanel:
            self.change = nn.Sequential(nn.Conv2d(inputchanel,outchanel,stride=stride,kernel_size=3,padding=0),
                                        nn.BatchNorm2d(outchanel),
                                        nn.ReLU(inplace=True))
    def forward(self,x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        print(out.shape)
        print(self.change(x).shape)

        x = self.change(x)+out
        x = self.relu(x)
        return x

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,stride=2,kernel_size=3,padding=0),
                                   nn.BatchNorm2d(16)
                                   )
        self.block1 = bottle(16,32,stride=2)
        self.block2 = bottle(32,64,stride=3)
        self.block3 = bottle(64,128,stride=2)
        self.block4 = bottle(128,256,stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(256*3*3,5)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # print(x.shape)
        x = x.view(x.shape[0],-1)
        # print(x.shape)
        x = self.linear(x)
        return x


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
def main():
    model = resnet18()
    # b = bottle(64,64)
    # bottle = bottle(64,128)
    # print(model)
    # x = torch.randn(16,3,224,224)
    # out = model(x)
    # print(out.shape)
    # p = sum(map(lambda p:p.numel(),model.parameters()))
    # print('parameters size:',p)
    # x = b(x)
    # print(x.shape)
if __name__ == '__main__':
    model = resnet18()
    print_network(model)
    # main()
