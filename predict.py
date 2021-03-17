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


# Prepare Test Data Generator
test_files = list_files(test_dir)

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
    num_workers=0,
)

model = UNet(1, 1).cuda()
model.load_state_dict(torch.load(model_file)["state_dict"])
model.eval()
# <---------------Test Loop---------------------->
with torch.no_grad():
    pbar = tqdm(test_loader, desc="description")
    test_dataset.reset_counters()
    for image in pbar:
        image = torch.autograd.Variable(image).cuda()
        output = model(image)
        output = torch.sigmoid(output)
        output = torch.round(output)

        # TODO:
        # save the images
