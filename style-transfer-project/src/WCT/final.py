#Final WCT Code that needed to run
import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from loader import Dataset
from utils import *



parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath',default='images/content',help='path to train')
parser.add_argument('--stylePath',default='images/style',help='path to train')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7', help='Path to the decoder1')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='samples/', help='folder to output images')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")
parser.add_argument('--level', type = int, default = -1, help="Multi = -1, else for specific level use level = {1,2,3,4,5}")
parser.add_argument('--do_patches', type = bool, default = False)
parser.add_argument('--kernel_size', type = int, default = 512)
parser.add_argument('--stride', type = int, default = 512)

args = parser.parse_args()

os.makedirs(args.outf)


# Data loading code
dataset = Dataset(args.contentPath,args.stylePath,args.fineSize, args.do_patches, args.kernel_size)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

wct = WCT(args)

def squeeze(a):
    return torch.squeeze(a.data.cpu(),0)

def styleTransfer(contentImg,styleImg,imname,csF, level = -1):
	# print("level= ",level)
    if(level == -1):
    	# print("Multi Level Style Transfer")
        sF5 = wct.e5(styleImg)
        cF5 = wct.e5(contentImg)
        sF5 = squeeze(sF5)
        # if(not args.do_patches):
        cF5 = squeeze(cF5)
        csF5 = wct.transform(cF5,sF5,csF,args.alpha)
        Im5 = wct.d5(csF5)

        sF4 = wct.e4(styleImg)
        cF4 = wct.e4(Im5)
        sF4 = squeeze(sF4)
        cF4 = squeeze(cF4)
        csF4 = wct.transform(cF4,sF4,csF,args.alpha)
        Im4 = wct.d4(csF4)

        sF3 = wct.e3(styleImg)
        cF3 = wct.e3(Im4)
        sF3 = squeeze(sF3)
        cF3 = squeeze(cF3)
        csF3 = wct.transform(cF3,sF3,csF,args.alpha)
        Im3 = wct.d3(csF3)

        sF2 = wct.e2(styleImg)
        cF2 = wct.e2(Im3)
        sF2 = squeeze(sF2)
        cF2 = squeeze(cF2)
        csF2 = wct.transform(cF2,sF2,csF,args.alpha)
        Im2 = wct.d2(csF2)

        sF1 = wct.e1(styleImg)
        cF1 = wct.e1(Im2)
        sF1 = squeeze(sF1)
        cF1 = squeeze(cF1)
        csF1 = wct.transform(cF1,sF1,csF,args.alpha)
        Im1 = wct.d1(csF1)

        vutils.save_image(Im5.data.cpu().float(),os.path.join(args.outf,"Level5_"+imname))
        vutils.save_image(Im4.data.cpu().float(),os.path.join(args.outf,"Level4_"+imname))
        vutils.save_image(Im3.data.cpu().float(),os.path.join(args.outf,"Level3_"+imname))
        vutils.save_image(Im2.data.cpu().float(),os.path.join(args.outf,"Level2_"+imname))
        vutils.save_image(Im1.data.cpu().float(),os.path.join(args.outf,"Level1_"+imname))
        print("MultiLevel")
        # save_image has this wired design to pad images with 4 pixels at default.
    elif (level == 1):
    	# print(level)
        sF1 = wct.e1(styleImg)
        cF1 = wct.e1(contentImg)
        # print(cF1.shape)
        sF1 = squeeze(sF1)
        cF1 = squeeze(cF1)
        csF1 = wct.transform(cF1,sF1,csF,args.alpha)
        Im1 = wct.d1(csF1)
        # print(level)
    elif (level == 2):
    	# print("Computation in Level 2")
        sF2 = wct.e2(styleImg)
        cF2 = wct.e2(contentImg)
        sF2 = squeeze(sF2)
        cF2 = squeeze(cF2)
        csF2 = wct.transform(cF2,sF2,csF,args.alpha)
        Im1 = wct.d2(csF2)
    elif (level == 3):
    	# print("Computation in Level 3")
    	sF3 = wct.e3(styleImg)
    	cF3 = wct.e3(contentImg)
    	sF3 = squeeze(sF3)
    	cF3 = squeeze(cF3)
    	csF3 = wct.transform(cF3,sF3,csF,args.alpha)
    	Im1 = wct.d3(csF3)
    elif (level == 4):    	
    	# print("Computation in Level 4")
        sF4 = wct.e4(styleImg)
        cF4 = wct.e4(contentImg)
        sF4 = squeeze(sF4)
        cF4 = squeeze(cF4)
        csF4 = wct.transform(cF4,sF4,csF,args.alpha)
        Im1 = wct.d4(csF4)    
        # whiten_Decode = wct.d4(whiten_cF)
        # vutils.save_image(Im1.data.cpu().float(),os.path.join(args.outf,"Level4_whiten_Decoder"imname))    

    elif (level == 5):
    	# print("Computation in Level 5")
        sF5 = wct.e5(styleImg)
        cF5 = wct.e5(contentImg)
        sF5 = squeeze(sF5)
        cF5 = squeeze(cF5)
        csF5 = wct.transform(cF5,sF5,csF,args.alpha)
        Im1 = wct.d5(csF5)

    vutils.save_image(Im1.data.cpu().float(),os.path.join(args.outf,imname))
    # return Im1

cImg = torch.Tensor()
sImg = torch.Tensor()
csF = torch.Tensor()
csF = Variable(csF)
if(args.cuda):
    cImg = cImg.cuda(args.gpu)
    sImg = sImg.cuda(args.gpu)
    csF = csF.cuda(args.gpu)
    wct.cuda(args.gpu)
for contentImg,styleImg,imname in loader:
    imname = imname[0]
    
    print("Here level is =",args.level)
    print('Computing the Image [PROCESSING]:  ' + imname)
    if (args.cuda):
        contentImg = contentImg.cuda(args.gpu)
        styleImg = styleImg.cuda(args.gpu)
    cImg = Variable(contentImg,volatile=True)
    sImg = Variable(styleImg,volatile=True)

    styleTransfer(cImg,sImg,imname,csF,args.level)
