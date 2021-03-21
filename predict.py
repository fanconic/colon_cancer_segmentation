import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
import torchvision
import torchvision.transforms as tfms
import torch
import torch.utils.data as data
from src.data.loader import CustomTestLoader
from src.data.loader import test_collate
from src.model.unet import UNet
from settings import (
    test_dir,
    predictions_dir,
    img_size,
    test_batch_size,
    model_file,
    chkpoint_file,
    seed,
)
from src.data.preprocessing import normalize, hounsfield_clip
import src
from src.utils.utils import list_files


# Prepare Test Data Generator
test_files = sorted(list_files(test_dir))

test_dataset = CustomTestLoader(
    test_dir,
    test_files,
    transforms=tfms.Compose(
        [
            tfms.ToTensor(),
            tfms.Lambda(hounsfield_clip),
            tfms.Lambda(normalize),
        ]
    ),
)


test_loader = data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=test_batch_size,
    collate_fn=test_collate,
    num_workers=0,
)

model = UNet(1, 1).cuda()
model.load_state_dict(torch.load(model_file)["state_dict"])
model.eval()
# <---------------Test Loop---------------------->
with torch.no_grad():
    pbar = tqdm(test_loader, desc="description")
    for i, image in enumerate(pbar):
        image = torch.autograd.Variable(image).cuda()

        # check that test_file[i] is equal to loaded file
        img_name = test_files[i]
        loaded_file = nib.load(
            os.path.join(test_dir, img_name)
        ).get_fdata()  # 512x512xDepth
        assert image.shape[0] == loaded_file.shape[2]
        image_split = torch.tensor_split(image, image.shape[0])

        # predict 2D slices since 3D too large for GPU
        output_ls = []
        for split in image_split:
            output = model(split)
            median = cv2.medianBlur(output[0][0].cpu().numpy(), 5)
            output = torch.Tensor([median]).cuda()
            output = torch.sigmoid(output)
            output = output.round()
            output_ls.append(output)

        output = torch.stack(output_ls)
        output = torch.squeeze(output)  # reducing from depth x 1 x 1 x height x width

        # save images
        prediction_filename = os.path.join(
            predictions_dir, "prediction_" + test_files[i]
        )
        with open(prediction_filename + ".pickle", "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # to test whether saving works
        with open(prediction_filename + ".pickle", "rb") as handle:
            unserialized_data = pickle.load(handle)
        assert torch.equal(output, unserialized_data)
