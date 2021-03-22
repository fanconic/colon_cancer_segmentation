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
        downsample=False,  # use all positive images, and some negative ones
        upsample=False,  # use repeat positive samples until balanced
        shuffle=False,
    ):
        self.root_dir = root_dir
        self.seg_dir = seg_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.files = files
        self.labels = labels
        self.downsample = downsample
        self.upsample = upsample
        self.shuffle = shuffle
        self.counter = 0
        self.index_offset = 0
        self.length = None
        print("Number of files: ", len(self.files))

        if self.shuffle:
            files, labels = sklearn.utils.shuffle(files, labels, random_state=seed)

        # Extract the first file
        self.current_img_nib, self.current_label_nib = self.load_nibs(
            self.files[self.counter], self.labels[self.counter]
        )
        self.counter += 1

    def __len__(self):
        """
        Computing the depthof the given image object
        Returns: depth of image object

        """
        if self.length is None:
            len = 0
            print("Calculating Data Set, this might take a while...")
            for x in self.files:
                img = nib.load(os.path.join(self.seg_dir, x))
                if self.downsample:
                    img = img.get_fdata()
                    non_blanks = (img != 0).any((0, 1))
                    img = img[:, :, non_blanks]
                    len += img.shape[2] + 10
                elif self.upsample:
                    img = img.get_fdata()
                    non_blanks = (img != 0).any((0, 1))
                    img = img[:, :, ~non_blanks]
                    len += img.shape[2] * 2
                else:
                    len += img.shape[2]
                del img
            self.length = len
            return len
        else:
            return self.length

    def __getitem__(self, idx):
        """
        Loading a new image with corresponding labels
        Args:
            idx: index of image
        Returns: image object and corresponding label object
        """

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
        """
        Load a new nib file, optionally also skips all the blank labels
        Args:
            x_file: path to the image
            y_file: path to mask
        Returns newly loaded NIB files
        """
        img = nib.load(os.path.join(self.root_dir, x_file)).get_fdata()
        img = np.float32(img)
        label = nib.load(os.path.join(self.seg_dir, y_file)).get_fdata()
        label = np.float32(label)

        # Only use depth channels which contain a positive label and 10 negative ones.
        # downsample negatives
        if self.downsample:
            if self.shuffle:
                p = np.random.RandomState(seed=seed).permutation(img.shape[2])
                img = img[:, :, p]
                label = label[:, :, p]

            non_blanks = (label != 0).any((0, 1))
            label_pos = label[:, :, non_blanks]
            img_pos = img[:, :, non_blanks]
            label_neg = label[:, :, ~non_blanks]
            img_neg = img[:, :, ~non_blanks]

            img = np.concatenate([img_neg[:, :, :10], img_pos], axis=2)
            label = np.concatenate([label_neg[:, :, :10], label_pos], axis=2)

        # Upsample positives
        if self.upsample:
            non_blanks = (label != 0).any((0, 1))
            label_pos = label[:, :, non_blanks]
            img_pos = img[:, :, non_blanks]
            label_neg = label[:, :, ~non_blanks]
            img_neg = img[:, :, ~non_blanks]
            n_pos = label_pos.shape[2]
            n_neg = label_neg.shape[2]
            if n_neg > n_pos:
                label_pos_new = np.transpose(
                    np.array(
                        [label_pos[:, :, (i % n_pos)] for i in range(n_neg - n_pos)]
                    ),
                    axes=(1, 2, 0),
                )
                img_pos_new = np.transpose(
                    np.array(
                        [img_pos[:, :, (i % n_pos)] for i in range(n_neg - n_pos)]
                    ),
                    axes=(1, 2, 0),
                )
                img = np.concatenate([img, img_pos_new], axis=2)
                label = np.concatenate([label, label_pos_new], axis=2)

        # shuffle the depth
        if self.shuffle:
            p = np.random.RandomState(seed=seed).permutation(img.shape[2])
            img = img[:, :, p]
            label = label[:, :, p]

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
    Custom Test Loader for CT images, such that these can be processed directly out of memory.
    Returns: image object
    """

    def __init__(
        self,
        root_dir,
        files,
        transforms=None,
    ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.files = files
        print("Number of files: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img = nib.load(os.path.join(self.root_dir, img_name)).get_fdata()
        img = np.float32(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img


def test_collate(batch):
    """
    custom collate function for 3d images of variable depth for testing
    Params:
        batch: is the next batch which should be processed
    Returns:
        list containing the data
    """
    data = [item for item in batch]
    data = torch.stack(list(itertools.chain(*data))).unsqueeze(1)

    return data


class CustomValidLoader(data.Dataset):
    """
    Custom Data Loader for CT iamges, such that these can be processed directly
    out of memory.
    Returns: image object and corresponding label object
    """

    def __init__(
        self,
        root_dir,
        seg_dir,
        files,
        labels,
        transforms=None,
        target_transforms=None,
    ):
        self.root_dir = root_dir
        self.seg_dir = seg_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.files = files
        self.lables = labels
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

        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transforms is not None:
            label = self.target_transforms(label)

        return img, label


def valid_collate(batch):
    """
    custom collate function for 3d images and labels of variable depth
    Params:
        batch: is the next batch which should be processed
    Returns:
        list containing the data and the target
    """
    data = [item[0] for item in batch]
    data = torch.stack(list(itertools.chain(*data))).unsqueeze(1)
    target = [item[1] for item in batch]
    target = torch.stack(list(itertools.chain(*target)))

    return [data, target]
