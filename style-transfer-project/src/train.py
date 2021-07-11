import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.models.vgg as models
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from tqdm import tqdm
debug = False

def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

device = get_device()

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).cuda()
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).cuda()
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
        return loss

def train_epoch(loader, model, optimizer = None, loss_func_mse = None, loss_func_perp = None, epoch = None):
    
    losses = []
    with tqdm(loader, unit="batch") as tepoch:
        
        for idx,im in enumerate(tepoch):
            tepoch.set_description(f"Epoch (Train) {epoch}")
            im = im.to(device)
            out = model(im)
            loss = loss_func_mse(out,im) + loss_func_perp(out,im)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if(debug and idx == 5):break
            
            tepoch.set_postfix(loss=loss.item())        
                
    print('\n---Epoch (Train) {0} Mean Loss [{1:.8f}]---'.format(epoch, np.mean(losses)))
    

def eval_epoch(loader, model, loss_func_mse = None, loss_func_perp = None,epoch = None):
    
    losses = []
    with torch.no_grad():
        with tqdm(loader, unit="batch") as tepoch:
            for idx,im in enumerate(tepoch):
                tepoch.set_description(f"Epoch (Val) {epoch}")
                im = im.to(device)
                out = model(im)
                loss = loss_func_mse(out,im)
                
                losses.append(loss.item())
                
                if(debug and idx == 10):break
                tepoch.set_postfix(loss=loss.item())
            
        print('\n---Epoch (Val) {0} Mean Loss [{1:.8f}]---'.format(epoch, np.mean(losses)))
        return np.mean(losses)

def train(loaders, model, eval_only = True, lr = 1e-4, decay = 0.0004, save_model = True, num_epochs = 20, patience = 10, model_dict_path = "../../ae_model.pth"):
    prev_val_loss = 1e10
    to_optim = [{'params':model.parameters(),'lr':lr,'weight_decay':decay}]
    
    optimizer = optim.Adam(to_optim)  
    loss_func_mse = nn.MSELoss()
    loss_func_perp = VGGPerceptualLoss()
    
    count = 0
    if (not eval_only):
        for epoch in range(num_epochs):
            model.train()
            train_epoch(loaders['train'], model, optimizer, loss_func_mse, loss_func_perp, epoch)
            model.eval()
            val_loss = eval_epoch(loaders['val'], model, loss_func_mse, loss_func_perp, epoch)
            count = count + 1
            if(save_model and val_loss < prev_val_loss):
                torch.save(model.state_dict(), model_dict_path)
                prev_val_loss = val_loss
                count = 0
            if count == patience:
                break
    else:
        val_loss = eval_epoch(loaders['val'], model, loss_func_mse, loss_func_perp, 0)      

        

    
