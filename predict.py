import os
import numpy as np
import nibabel as nib
import cv2
import pickle
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
    input_channels,
    output_channels,
    ensemble,
    ensemble_model_1,
    ensemble_model_2,
    ensemble_model_3,
    ensemble_model_4,
    ensemble_model_5,
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

if ensemble:
    # To make prediction more stables take models from the best 5 epochs
    # model 1
    model_1 = UNet(input_channels, output_channels).cuda()
    model_1.load_state_dict(torch.load(ensemble_model_1)["state_dict"])
    model_1.eval()
    # model 2
    model_2 = UNet(input_channels, output_channels).cuda()
    model_2.load_state_dict(torch.load(ensemble_model_2)["state_dict"])
    model_2.eval()
    # model 3
    model_3 = UNet(input_channels, output_channels).cuda()
    model_3.load_state_dict(torch.load(ensemble_model_3)["state_dict"])
    model_3.eval()
    # model 4
    model_4 = UNet(input_channels, output_channels).cuda()
    model_4.load_state_dict(torch.load(ensemble_model_4)["state_dict"])
    model_4.eval()
    # model 5
    model_5 = UNet(input_channels, output_channels).cuda()
    model_5.load_state_dict(torch.load(ensemble_model_5)["state_dict"])
    model_5.eval()

else:
    # simple model prediction with the best model
    model = UNet(input_channels, output_channels).cuda()
    model.load_state_dict(torch.load(model_file)["state_dict"])
    model.eval()

# <---------------Test Loop---------------------->
with torch.no_grad():
    for i, image in enumerate(test_loader):
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
            if ensemble:
                output1 = model_1(split)
                output1 = torch.sigmoid(output1)
                output2 = model_2(split)
                output2 = torch.sigmoid(output2)
                output3 = model_3(split)
                output3 = torch.sigmoid(output3)
                output4 = model_4(split)
                output4 = torch.sigmoid(output4)
                output5 = model_5(split)
                output5 = torch.sigmoid(output5)
                output = (output1 + output2 + output3 + output4 + output5) / 5

            else:
                output = model(split)
                output = torch.sigmoid(output)
            output = output.round().type(torch.uint8)
            median = cv2.medianBlur(output[0][0].cpu().numpy(), 11)
            output = torch.Tensor([median]).cuda()
            output_ls.append(output)

        output = torch.stack(output_ls)
        output = torch.squeeze(output)  # reducing from depth x 1 x 1 x height x width
        output = output.permute(1, 2, 0) # rearranging to height x width x depth

        # save images
        prediction_filename = os.path.join(
            predictions_dir, "prediction_" + test_files[i]
        )
        with open(prediction_filename.split(".")[0] + ".pkl", "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
