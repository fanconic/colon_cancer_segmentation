import torch
import torch.nn.functional as F
import torch.nn as nn


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        """
        Computing the IOU (Intersection over Union) as described in tutorial session
        Args:
            inputs: predicted classificaiton (primary, background)
            targets: underlying truth (0,1)
            smooth: smoothing factor
        Returns:
            IOU score
        """
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
        """
        Computing the IOU (Intersection over Union) with rounded inputs (0,1)
        Args:
            inputs: predicted and rounded classificaiton (primary, background)
            targets: underlying truth (0,1)
            smooth: smoothing factor
        Returns:
            Threshold IOU score
        """

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors

        inputs = inputs.view(-1)
        inputs = inputs.round()  # Outputs are rounded to 0 or 1
        targets = targets.view(-1)

        i = (inputs == 1) & (targets == 1)
        u = (inputs == 1) | (targets == 1)

        nominator = torch.sum(i) + smooth
        denominator = torch.sum(u) + smooth

        IoU = nominator / denominator

        return IoU


class IoU_3D(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_3D, self).__init__()

    def forward(self, inputs, targets):
        """
        Computing the 3D IOU (Intersection over Union) as described on Piazza
        Args:
            inputs: predicted classificaiton (primary, background)
            targets: underlying truth (0,1)
        Returns:
            3D IOU score
        """
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # Round and squeeze
        inputs = torch.squeeze(inputs)
        inputs = inputs.round()

        assert inputs.ndim == 3 and targets.ndim == 3
        i = (inputs == 1) & (targets == 1)
        u = (inputs == 1) | (targets == 1)

        iou = torch.sum(i) / torch.sum(u)
        return iou
