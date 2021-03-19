import os
import numpy as np
import nibabel as nib
import random
import torch
import torch.utils.data as data
import itertools
from src.utils.utils import list_files
import sklearn.utils
from settings import seed


class CustomDataLoader(data.Dataset):
    """
    Custom Data Loader for CT iamges, such that these can be processed directly
    out of memory.
    """

    def __init__(
        self,
        root_dir,
        seg_dir,
        files,
        labels,
        transforms=None,
        target_transforms=None,
        skip_blank=False,
        shuffle=False,
    ):
        self.root_dir = root_dir
        self.seg_dir = seg_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.files = files
        self.labels = labels
        self.skip_blank = skip_blank
        self.shuffle = shuffle
        self.counter = 0
        self.index_offset = 0
        print("Number of files: ", len(self.files))

        if self.shuffle:
            files, labels = sklearn.utils.shuffle(files, labels, random_state=seed)

        # Extract the first file
        self.current_img_nib, self.current_label_nib = self.load_nibs(
            self.files[self.counter], self.labels[self.counter]
        )
        self.counter += 1

    def __len__(self):
        len = 0
        for x in self.files:
            img = nib.load(os.path.join(self.root_dir, x))
            len += img.shape[2]
            del img
        return len

    def __getitem__(self, idx):

        # Load new file if index is out of range
        if (idx - self.index_offset) >= self.current_img_nib.shape[2]:
            self.index_offset += self.current_img_nib.shape[2]
            self.current_img_nib, self.current_label_nib = self.load_nibs(
                self.files[self.counter], self.labels[self.counter]
            )
            self.counter += 1

        # Extract image and label
        img = self.current_img_nib[:, :, idx - self.index_offset]
        label = self.current_label_nib[:, :, idx - self.index_offset]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.transforms is not None:
            img = self.transforms(img)

        random.seed(seed)  # apply this seed to target transforms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.target_transforms is not None:
            label = self.target_transforms(label)

        return img, label[0]

    def load_nibs(self, x_file, y_file):
        """Load a new nib file, optionally also skips all the blank labels
        Args:
            x_file: path to the image
            y_file: path to mask
        Returns newly loaded NIB files
        """
        img = nib.load(os.path.join(self.root_dir, x_file)).get_fdata()
        img = np.float32(img)
        label = nib.load(os.path.join(self.seg_dir, y_file)).get_fdata()
        label = np.float32(label)

        # Only use depth channels which contain a positive label
        if self.skip_blank:
            non_blanks = (label != 0).any((0, 1))
            label = label[:, :, non_blanks]
            img = img[:, :, non_blanks]

        # shuffle the depth
        if self.shuffle:
            p = np.random.RandomState(seed=seed).permutation(img.shape[2])
            img = img[:,:,p]
            label = label[:,:,p]

        return img, label

    def reset_counters(self):
        """Resets the counters"""
        self.counter = 0
        self.index_offset = 0
        self.current_img_nib, self.current_label_nib = self.load_nibs(
            self.files[self.counter], self.labels[self.counter]
        )
        self.counter += 1


class CustomTestLoader(data.Dataset):
    """
    Custom Test Loader for CT iamges, such that these can be processed directly
    out of memory.
    """

    def __init__(
        self,
        root_dir,
        files,
        transforms=None,
    ):
        self.root_dir = root_dir
        self.files = files
        self.transforms = transforms
        self.counter = 0
        self.index_offset = 0
        print("Number of files: ", len(self.files))

        # Extract the first file
        self.current_img_nib = self.load_nibs(self.files[self.counter])
        self.counter += 1

    def __len__(self):
        len = 0
        for x in self.files:
            img = nib.load(os.path.join(self.root_dir, x))
            len += img.shape[2]
            del img
        return len

    def __getitem__(self, idx):

        # Load new file if index is out of range
        if (idx - self.index_offset) >= self.current_img_nib.shape[2]:
            self.index_offset += self.current_img_nib.shape[2]
            self.current_img_nib = self.load_nibs(self.files[self.counter])
            self.counter += 1

        # Extract image and label
        img = self.current_img_nib[:, :, idx - self.index_offset]

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def load_nibs(self, x_file):
        """Load a new nib file
        Args:
            x_file: path to the image
        Returns newly loaded NIB files
        """
        img = nib.load(os.path.join(self.root_dir, x_file)).get_fdata()
        img = np.float32(img)

        return img

    def reset_counters(self):
        """Resets the counters"""
        self.counter = 0
        self.index_offset = 0
        self.current_img_nib = self.load_nibs(self.files[self.counter])
        self.counter += 1