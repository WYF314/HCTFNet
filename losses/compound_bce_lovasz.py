from losses.bce import *
from losses.lovasz_softmax import *


class BceLovaszLoss(nn.Module):
    def __init__(self):
        super(BceLovaszLoss, self).__init__()
        self.bce = BceLoss()
        
    def forward(self, logits, true):
        weight = 0.7
        ## logits是model输出图像,true是label
        self.lovasz = lovasz_softmax(F.softmax(logits, dim=1), true, ignore=255)
        return weight*self.bce(logits, true) + (1-weight)*self.lovasz
