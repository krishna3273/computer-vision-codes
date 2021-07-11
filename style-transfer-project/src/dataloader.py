import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms

class simpleCOCO(data.Dataset):
    
    # initialise function of class
    def __init__(self, root, filenames):
        # the data directory 
        self.root = root
        # the list of filename
        self.filenames = filenames

    # obtain the sample with the given index
    def __getitem__(self, index):
        # obtain filenames from list
        image_filename = self.filenames[index]
        # Load data and label
        image = Image.open(os.path.join(self.root, image_filename)).convert('RGB')
        
        # output of Dataset must be tensor
        trans=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        image=trans(image)
        return image
    
    # the total number of samples (optional)
    def __len__(self):
        return len(self.filenames)

def create_dataloaders(train_images_root, val_images_root, batch_size = 32, num_workers = 16):

    train_filenames = os.listdir(train_images_root)
    train_coco_data = simpleCOCO(train_images_root,train_filenames)

    val_filenames = os.listdir(val_images_root)
    val_coco_data = simpleCOCO(val_images_root,val_filenames)

    train_loader = data.DataLoader(train_coco_data, batch_size = batch_size, num_workers = num_workers, shuffle = True, drop_last = True)
    val_loader = data.DataLoader(val_coco_data, batch_size = batch_size, num_workers = num_workers, shuffle = True, drop_last = True)

    return {'train': train_loader, 'val': val_loader}
