import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import numpy as np

import time
import os
import copy


def conv_block(s1,s2,s3):
    return nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(s1,s2,s3),
            nn.ReLU()
        )

class model_selector(nn.Module):
    def __init__(self, weights_path,layer = 5, pretrained = True, train_decoder = False):
        super(model_selector, self).__init__()
        self.num_layer = layer
        vgg19 = models.vgg19(pretrained=pretrained)

        features = list(vgg19.features)
        
        if(self.num_layer == 1):
            self.encoder = nn.Sequential(*features[:4])
            self.decoder = nn.Sequential( # Sequential,
	                            nn.ReflectionPad2d((1, 1, 1, 1)),
	                            nn.Conv2d(64,3,(3, 3)),
                                )
        elif(self.num_layer == 2):
            self.encoder = nn.Sequential(*features[:9])
            self.decoder = nn.Sequential( # Sequential,
                                conv_block(128,64,(3,3)),
                                nn.UpsamplingNearest2d(scale_factor=2),
                                conv_block(64,64,(3,3)),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(64,3,(3, 3)),
                            )
        elif(self.num_layer == 3):
            self.encoder = nn.Sequential(*features[:18])
            self.decoder = nn.Sequential( # Sequential,
                    conv_block(256,128,(3,3)),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv_block(128,128,(3,3)),
                    conv_block(128,64,(3,3)),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv_block(64,64,(3,3)),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,3,(3, 3)),
                )
        elif(self.num_layer == 4):
            self.encoder = nn.Sequential(*features[:27])
            self.decoder = nn.Sequential( # Sequential,
                    conv_block(512,256,(3,3)),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv_block(256,256,(3,3)),
                    conv_block(256,256,(3,3)),
                    conv_block(256,256,(3,3)),
                    conv_block(256,128,(3,3)),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv_block(128,128,(3,3)),
                    conv_block(128,64,(3,3)),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv_block(64,64,(3,3)),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,3,(3, 3)),
                )
        elif(self.num_layer == 5):
            self.encoder = nn.Sequential(*features[:36])
            self.decoder = nn.Sequential( # Sequential,
                    conv_block(512,512,(3,3)),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv_block(512,512,(3,3)),
                    conv_block(512,512,(3,3)),
                    conv_block(512,512,(3,3)),
                    conv_block(512,256,(3,3)),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv_block(256,256,(3,3)),
                    conv_block(256,256,(3,3)),
                    conv_block(256,256,(3,3)),
                    conv_block(256,128,(3,3)),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv_block(128,128,(3,3)),
                    conv_block(128,64,(3,3)),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    conv_block(64,64,(3,3)),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,3,(3, 3)),
                )
    
        if(train_decoder):
            for param in encoder.parameters():
                param.requires_grad = False
        
        else:
            model_dict_path = os.path.join(weights_path,'feature_invertor_conv' + str(self.num_layer) + '_1.pth')
            self.decoder.load_state_dict(torch.load(model_dict_path))

    def forward(self,x):
        return self.decoder(self.encoder(x))