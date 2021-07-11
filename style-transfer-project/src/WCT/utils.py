#Data Loader for VGG
import torch
from torch.utils.serialization import load_lua
from model import decoder1,decoder2,decoder3,decoder4,decoder5,encoder1,encoder2,encoder3,encoder4,encoder5
import torchvision.transforms.functional as F


class WCT(torch.nn.Module):
    def __init__(self,args):
        super(WCT, self).__init__()
        # load pre-trained network
        vgg1 = load_lua(args.vgg1)
        decoder1_torch = load_lua(args.decoder1)
        vgg2 = load_lua(args.vgg2)
        decoder2_torch = load_lua(args.decoder2)
        vgg3 = load_lua(args.vgg3)
        decoder3_torch = load_lua(args.decoder3)
        vgg4 = load_lua(args.vgg4)
        decoder4_torch = load_lua(args.decoder4)
        vgg5 = load_lua(args.vgg5)
        decoder5_torch = load_lua(args.decoder5)

        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)
        self.e2 = encoder2(vgg2)
        self.d2 = decoder2(decoder2_torch)
        self.e3 = encoder3(vgg3)
        self.d3 = decoder3(decoder3_torch)
        self.e4 = encoder4(vgg4)
        self.d4 = decoder4(decoder4_torch)
        self.e5 = encoder5(vgg5)
        self.d5 = decoder5(decoder5_torch)

        self.args = args

    def whiten_and_color(self,cF,sF):
        cFSize = cF.shape
        c_mean = torch.mean(cF,1)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean
        num_dim=len(cF.shape)
        cF_t=torch.transpose(cF,num_dim-2,num_dim-1)
        k_c = cFSize[0]
        contentConv = torch.mm(cF,cF_t)/(cFSize[1]-1) + torch.eye(k_c,dtype=torch.double)
        c_u,c_e,c_v = torch.svd(contentConv,some=False)

        i=0
        while i<k_c:
            if c_e[i] < 0.00001:
                k_c = i
                break
            i+=1            

        sFSize = sF.shape
        s_mean = torch.mean(sF,1)
        num_dim=len(sFSize)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        sF_t=torch.transpose(sF,num_dim-2,num_dim-1)
        styleConv = torch.mm(sF,sF_t)/(sFSize[1]-1)
        s_u,s_e,s_v = torch.svd(styleConv,some=False)

        k_s = sFSize[0]
        i=0
        while i<sFSize[0]:
            if s_e[i] < 0.00001:
                k_s = i
                break
            i+=1

        c_d = (c_e[0:k_c])**-0.5
        eig_c=c_v[:,0:k_c]
        whiten_cF = torch.mm(torch.mm(torch.mm(eig_c,torch.diag(c_d)),(eig_c.t())),cF)

        s_d = (s_e[0:k_s])**0.5
        eig_v=s_v[:,0:k_s]
        targetFeature = torch.mm(torch.mm(torch.mm(eig_v,torch.diag(s_d)),(eig_v.t())),whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature,whiten_cF

    def transform(self,cF,sF,csF,alpha,return_whiten=False):
        cF = cF.to(torch.double)
        sF = sF.to(torch.double)
        
        C,W,H = cF.size()
        _,W1,H1 = sF.size()
        targetFeature,whiten_cF = self.whiten_and_color(cF.reshape((C,-1)),sF.reshape((C,-1)))
        targetFeature = targetFeature.view_as(cF)
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.to(torch.float).unsqueeze(0)
        csF.data.resize_(ccsF.shape).copy_(ccsF)
        return csF,whiten_cF if return_whiten==True else csF
