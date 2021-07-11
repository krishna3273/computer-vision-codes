from model import model_selector
from dataloader import create_dataloaders
from train import train
import torch
import torch.nn as nn

def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

# def main(pathname, model_dict_path = "model.pth",clf = False):

train_pathname = "../train2017/"
val_pathname = "../val2017/"
weight_path = "./Model_Weights"
device = get_device()
loaders = create_dataloaders(train_pathname, val_pathname)
model = model_selector(weight_path, layer = 5)
print ("---Loaded data and model---")
# if(torch.cuda.device_count() > 1):
#     model = nn.DataParallel(model)

model = model.to(device)

train(loaders, model)



