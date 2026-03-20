from losses.bce import *
from losses.dice import *
from losses.lovasz_softmax import *


class BceDiceLoss(nn.Module):
    def __init__(self):
        super(BceDiceLoss, self).__init__()
        self.bce = BceLoss()
        self.dice = DiceLoss()
        

    def forward(self, logits, true):
        weight_bce = 0.3
        weight_dice = 0.4
        weight_lovasz = 0.3
        return weight_bce * self.bce(logits, true) + weight_dice * self.dice(logits, true) + weight_lovasz * lovasz_softmax(F.softmax(logits, dim=1), true, ignore=255)
