import torch
import torchvision
from torch import nn
from torchvision.models import resnet18
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1,shape)
Trian_model = resnet18(pretrained=True)
Trian_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
print(Trian_model)
model = nn.Sequential(*list(Trian_model.children())[:-1],
                      Flatten(),
                      nn.Linear(512,5))
x = torch.randn(12,1,58,58)
x = model(x)
print(x.shape)