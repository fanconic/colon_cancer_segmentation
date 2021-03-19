import torch
import torch.nn.functional as F
import torch.nn as nn


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU


class Threshold_IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Threshold_IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        
        inputs = inputs.view(-1)
        inputs = inputs.round()  # Outputs are rounded to 0 or 1
        targets = targets.view(-1)

        i = (inputs==1) & (targets==1)
        u = (inputs==1) | (targets==1)

        nominator = torch.sum(i) + smooth
        denominator = torch.sum(u) + smooth

        IoU = nominator / denominator

        return IoU


class IoU_3D(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_3D, self).__init__()

    def forward(self, inputs, targets):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # Round and squeeze
        inputs = inputs.round()
        inputs = torch.squeeze(inputs)

        assert inputs.ndim==3 and targets.ndim==3
        i = (inputs==1) & (targets==1)
        u = (inputs==1) | (targets==1)

        print(i.shape, u.shape)

        iou = torch.sum(i) / torch.sum(u)
        return iou
