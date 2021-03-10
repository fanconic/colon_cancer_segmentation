import os
import numpy as np
import nibabel as nib
import random
import torch
import torch.utils.data as data
import itertools
from src.util.utils import list_files


class CustomDataLoader(data.Dataset):
    """
    Custom Data Loader for CT iamges, such that these can be processed directly
    out of memory.
    """

    def __init__(
        self,
        root_dir,
        seg_dir,
        transforms=None,
        target_transforms=None,
        skip_blank=False,
    ):
        self.root_dir = root_dir
        self.seg_dir = seg_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.files = list_files(self.root_dir)
        self.lables = list_files(self.seg_dir)
        self.skip_blank = skip_blank
        print("Number of files: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        label_name = self.lables[idx]
        img = nib.load(os.path.join(self.root_dir, img_name)).get_fdata()
        img = np.float32(img)
        label = nib.load(os.path.join(self.seg_dir, label_name)).get_fdata()
        label = np.float32(label)

        # Only use depth channels which contain a positive label
        if self.skip_blank:
            non_blanks = (label != 0).any((0, 1))
            label = label[:, :, non_blanks]
            img = img[:, :, non_blanks]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.transforms is not None:
            img = self.transforms(img)

        random.seed(seed)  # apply this seed to target tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.target_transforms is not None:
            label = self.target_transforms(label)

        return img, label


def custom_collate(batch):
    """
    custom collate function for 3d images of variable depth
    Params:
        batch: is the next batch which should be processed
    Returns:
        list containing the data and the target
    """
    data = [item[0] for item in batch]
    data = torch.stack(list(itertools.chain(*data))).unsqueeze(1)
    target = [item[1] for item in batch]
    target = torch.stack(list(itertools.chain(*target))).unsqueeze(1)

    return [data, target]


def custom_collate_permute(batch):
    """
    custom collate function for 3d images of variable depth, and also PERMUTES them
    Params:
        batch: is the next batch which should be processed
    Returns:
        list containing the data and the target
    """
    data = [item[0] for item in batch]
    data = torch.stack(list(itertools.chain(*data))).unsqueeze(1)
    perm = torch.randperm(data.size()[0])
    data = data[perm]
    target = [item[1] for item in batch]
    target = torch.stack(list(itertools.chain(*target))).unsqueeze(1)
    target = target[perm]

    return [data, target]