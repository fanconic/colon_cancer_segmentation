import os
import numpy as np
from tqdm import tqdm
import torchvision
import torchvision.transforms as tfms
import torch
import torch.utils.data as data
from src.data.loader import CustomTestLoader, test_collate
from src.model.unet import UNet
from settings import (
    test_dir,
    img_size,
    batch_size,
    model_file,
    chkpoint_file,
    seed,
)
from src.data.preprocessing import resize, normalize
import src
from src.utils.utils import list_files


test_files = list_files(test_dir)

# Prepare Test Data Generator
test_dataset = CustomTestLoader(
    test_dir,
    test_files,
    transforms=tfms.Compose(
        [
            tfms.ToTensor(),
            tfms.Lambda(lambda x: resize(x, size=img_size)),
            tfms.Lambda(normalize),
        ]
    ),
)


test_loader = data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=test_collate,
    num_workers=0,
)

# Load model
model = UNet(1, 1).cuda()
model.load_state_dict(torch.load(model_file)["state_dict"])
model.eval()


# <---------------Test Loop---------------------->
with torch.no_grad():
    pbar = tqdm(test_loader, desc="description")
    for image in pbar:
        image = torch.autograd.Variable(image).cuda()
        output = model(image)

        # TODO:
        # save the images
