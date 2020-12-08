import torch.nn as nn

__all__ = ['BNET2d']

"""
bnet as shown in paper
"""
class BNET2d(nn.BatchNorm2d):
    """
    To use, simply replace bn with BNET
    
    ::param width: number of input channels
    ::param k: kernel size of the transformation
    """
    def __init__(self, width, *args, k=3, **kwargs):
        super(BNET2d, self).__init__(width, *args, affine=False, **kwargs)
        self.bnconv = nn.Conv2d(width, width, k, padding=(k-1)//2, groups=width, bias=True)
        
    def forward(self, x):
        return self.bnconv(super(BNET2d, self).forward(x))