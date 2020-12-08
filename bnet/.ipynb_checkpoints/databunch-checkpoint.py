# NOTE: This DataBunch module only works for Imagenette at the moment
# I will update this for various datasets
import PIL
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from pathlib import Path
import random
from tqdm.notebook import tqdm as tqdm
import numpy as np

__all__ = ['default_tfms', 'ImageNette', 'DataBunch']

def default_tfms(size):
    tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    return tfms

class ImageNette(Dataset):
    def __init__(self, ROOT, train=True, shuffle=True, tfms=None):
        self._train_ = train
        self.tfms = default_tfms(size=128) if tfms is None else tfms
        self.ROOT = ROOT
        self.path = ROOT/'train' if train==True else ROOT/'val'
        
        self.n2c = {v:i for i,v in enumerate(os.listdir(self.path))}
        self.c2n = {v:k for k,v in self.n2c.items()}
        
        data = []
        for c in self.n2c.keys():
            p2fol = os.path.join(self.path, c)
            for f in os.listdir(p2fol):
                p2im = os.path.join(p2fol, f)
                data.append(p2im)
                
        self.data = data
        self.jpeg_filter()
        if shuffle: random.shuffle(self.data)
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        p2im = self.data[idx]
        im = PIL.Image.open(p2im)
        if self.tfms: im = self.tfms(im)
        y = self.get_cls(p2im)
        y = torch.Tensor([float(y)]).squeeze(0).long()
        return im, y
        
    def get_cls(self, p2im): 
        cname = p2im.split('\\')[3]
        return self.n2c[cname]
    
    def jpeg_filter(self):
        """
        Removing grayscale
        """
        print(f"Removing grayscale from: {'train' if self._train_ else 'valid'} dataset")
        keep = []
        n = len(self.data)
        for i in tqdm(range(n)):
            im = PIL.Image.open(self.data[i])
            nc = len(np.array(im).shape)
            if nc==3: keep.append(self.data[i])
                
        self.data = keep
        return self
    
class DataBunch:
    def __init__(self, root, bs=32, tfms=None, num_workers=0):
        """
        ::param root: pointing to imagenette folder
        """
        self.train_ds = ImageNette(root, train=True, tfms=tfms)
        self.valid_ds = ImageNette(root, train=False, tfms=tfms)
        
        self.train_dl = DataLoader(
            self.train_ds, batch_size=bs, num_workers=num_workers, shuffle=True)
        
        self.valid_dl = DataLoader(
            self.valid_ds, batch_size=bs, num_workers=num_workers, shuffle=False)