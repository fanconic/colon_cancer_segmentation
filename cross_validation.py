import os
import numpy as np
from tqdm import tqdm
import torchvision
import torchvision.transforms as tfms
import torch
import torch.utils.data as data
from src.data.loader import CustomDataLoader, CustomValidLoader, valid_collate
from settings import (
    train_dir,
    labels_dir,
    img_size,
    skip_empty,
    batch_size,
    input_channels,
    output_channels,
    model_file,
    chkpoint_file,
    learning_rate,
    num_epochs,
    train_val_splitting_ratio,
    seed,
    max_epochs_no_improve,
    shuffle_files,
    k_folds,
)
from src.data.preprocessing import resize, normalize, torch_equalize
from src.model.unet import UNet
import src
from src.model.losses import DiceLoss
from src.model.metrics import IoU, Threshold_IoU, IoU_3D
from sklearn.model_selection import train_test_split
from src.utils.utils import list_files
from sklearn.model_selection import KFold

# splitting data into train and val sets
files = sorted(list_files(train_dir))
lables = sorted(list_files(labels_dir))

# K-Fold Cross Validation
kfold = KFold(n_splits=k_folds, shuffle=True)

total_train_loss = []
total_train_score = []
total_train_score_round = []

total_valid_loss = []
total_valid_score = []
total_valid_score_round = []
total_valid_3d_score = []

for fold, (train_ids, dev_ids) in enumerate(kfold.split(files)):
    #train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
    #dev_sampler = torch.utils.data.SubsetRandomSampler(dev_ids)    

    print(train_ids)

    # Prepare Training Data Generator
    train_dataset = CustomDataLoader(
        train_dir,
        labels_dir,
        train_ids,
        train_ids,
        skip_blank=skip_empty,
        shuffle=shuffle_files,
        transforms=tfms.Compose(
            [
                tfms.ToTensor(),
                tfms.Lambda(lambda x: resize(x, size=img_size)),
                tfms.RandomRotation(5, fill=-1024),
                tfms.Lambda(normalize),
                tfms.Lambda(torch_equalize),
            ]
        ),
        target_transforms=tfms.Compose(
            [
                tfms.ToTensor(),
                tfms.Lambda(lambda x: resize(x, size=img_size)),
                tfms.RandomRotation(5, fill=0),
            ]
        ),
    )

    # Prepare Val Data Generator
    val_dataset = CustomValidLoader(
        train_dir,
        labels_dir,
        dev_ids,
        dev_ids,
        transforms=tfms.Compose(
            [
                tfms.ToTensor(),
                tfms.Lambda(lambda x: resize(x, size=img_size)),
                tfms.Lambda(normalize),
                tfms.Lambda(torch_equalize),
            ]
        ),
        target_transforms=tfms.Compose(
            [
                tfms.ToTensor(),
                tfms.Lambda(lambda x: resize(x, size=img_size)),
            ]
        )
    )

    # Create train and validation data loader
    train_loader = data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
    )

    val_loader = data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=valid_collate,
        num_workers=0,
    )

    # Define the model and optimizer
    model = UNet(input_channels, output_channels).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    # from engine import evaluate
    criterion = DiceLoss()
    accuracy_metric = IoU()
    threshold_metric = Threshold_IoU()
    iou_3d = IoU_3D()
    valid_loss_min = np.Inf


    # vars for early stopping
    epochs_no_improve = 0
    best_current_checkpoint = None
    best_current_checkpoint_file = None
    best_current_model_file = None

    losses_value = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        train_score = []
        train_score_round = []

        valid_loss = []
        valid_score = []
        valid_score_round = []
        valid_3d_score = []

        # <-----------Training Loop---------------------------->
        # reset the counters
        train_dataset.reset_counters()
        for x_train, y_train in train_loader:
            x_train = torch.autograd.Variable(x_train).cuda()
            y_train = torch.autograd.Variable(y_train).cuda()
            optimizer.zero_grad()
            output = model(x_train)
            # Loss
            loss = criterion(output, y_train)
            losses_value = loss.item()
            # Score
            score = accuracy_metric(output, y_train)
            score_t = threshold_metric(output, y_train)
            # Optimizing
            loss.backward()
            optimizer.step()
            # Logging
            train_loss.append(losses_value)
            train_score.append(score.item())
            train_score_round.append(score_t.item())

        # <---------------Validation Loop---------------------->
        model.eval()
        with torch.no_grad():
            for image, mask in val_loader:
                image = torch.autograd.Variable(image).cuda()
                mask = torch.autograd.Variable(mask).cuda()
                
                image_split = torch.tensor_split(image, image.shape[0])

                # predict 2D slices since 3D too large for GPU
                output_ls = []
                for split in image_split:
                    output = model(split)
                    output_ls.append(output)
                output = torch.stack(output_ls)

                loss = criterion(output, mask)
                losses_value = loss.item()
                ## Compute Accuracy Score
                score = accuracy_metric(output, mask)
                score_t = threshold_metric(output, mask)
                score_3d = iou_3d(output, mask)
                # logging
                valid_loss.append(losses_value)
                valid_score.append(score.item())
                valid_score_round.append(score_t.item())
                valid_3d_score.append(score_3d.item())

        total_train_loss.append(np.mean(train_loss))
        total_train_score.append(np.mean(train_score))
        total_train_score_round.append(np.mean(train_score_round))

        total_valid_loss.append(np.mean(valid_loss))
        total_valid_score.append(np.mean(valid_score))
        total_valid_score_round.append(np.mean(valid_score_round))
        total_valid_3d_score.append(np.mean(valid_3d_score))

        print(
            "\n###########Fold: {}, Train Loss: {}+-{}, Train IOU: {}+-{}, Train Threshold IoU: {}+-{}###########".format(
                fold,
                total_train_loss[-1],
                np.std(train_loss),
                total_train_score[-1],
                np.std(train_score),
                total_train_score_round[-1],
                np.std(train_score_round),
            )
        )

        print(
            "###########Fold: {}, Valid Loss: {}+-{}, Valid IOU: {}+-{}, Valid Threshold IoU: {}+-{}, Valid 3D IoU: {}+-{} ###########".format(
                fold,
                total_valid_loss[-1],
                np.std(valid_loss),
                total_valid_score[-1],
                np.std(valid_score),
                total_valid_score_round[-1],
                np.std(valid_score_round),
                total_valid_3d_score[-1],
                np.std(valid_3d_score),
            )
        )

        # Save best model Checkpoint
        # create checkpoint variable and add important data
        checkpoint = {
            "epoch": epoch + 1,
            "valid_loss_min": total_valid_loss[-1],
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # save checkpoint
        src.utils.utils.save_ckp(checkpoint, False, chkpoint_file + "epoch_{}.pt".format(epoch+1), model_file)

        if total_valid_loss[-1] <= valid_loss_min:
            # keeping track of current best model (for early stopping)
            epochs_no_improve = 0

        else:
            # epoch passed without improvement
            epochs_no_improve += 1

        # checking for early stopping
        if epochs_no_improve > max_epochs_no_improve:
            break

    print(
            "###########Fold: {}, Valid Loss Min: {}, Valid Loss {}+-{}, Valid IOU: {}+-{}, Valid Threshold IoU: {}+-{}, Valid 3D IoU: {}+-{} ###########".format(
                fold,
                valid_loss_min,
                total_valid_loss[-1],
                np.std(valid_loss),
                total_valid_score[-1],
                np.std(valid_score),
                total_valid_score_round[-1],
                np.std(valid_score_round),
                total_valid_3d_score[-1],
                np.std(valid_3d_score),
            )
        )

print(
        "########### Mean Total Dev Score: {} ###########".format(
                np.mean(total_valid_3d_score),
            )
    )
