import os
import numpy as np
from tqdm import tqdm
import torchvision
import torchvision.transforms as tfms
import torch
import torch.utils.data as data
from src.data.loader import CustomDataLoader, custom_collate_permute, custom_collate
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
)
from src.data.preprocessing import resize
from src.model.unet import UNet
import src

# Prepare Data Generator
full_dataset = CustomDataLoader(
    train_dir,
    labels_dir,
    transforms=tfms.Compose(
        [
            tfms.ToTensor(),
            tfms.Lambda(lambda x: resize(x, size=img_size)),
            tfms.RandomHorizontalFlip(),
            tfms.RandomVerticalFlip(),
            tfms.RandomRotation(45, fill=-1024),
        ]
    ),
    target_transforms=tfms.Compose(
        [
            tfms.ToTensor(),
            tfms.Lambda(lambda x: resize(x, size=img_size)),
            tfms.RandomHorizontalFlip(),
            tfms.RandomVerticalFlip(),
            tfms.RandomRotation(45, fill=0),
        ]
    ),
    skip_blank=skip_empty,
)

# Train Test Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

train_loader = data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=custom_collate_permute,
    num_workers=os.cpu_count(),
)

val_loader = data.DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=custom_collate,
    num_workers=os.cpu_count(),
)

# Define the model and optimizer
model = UNet(input_channels, output_channels).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
# from engine import evaluate
criterion = src.model.losses.DiceLoss()
accuracy_metric = src.model.metrics.IoU()
valid_loss_min = np.Inf


total_train_loss = []
total_train_score = []
total_valid_loss = []
total_valid_score = []

losses_value = 0
for epoch in range(num_epochs):

    train_loss = []
    train_score = []
    valid_loss = []
    valid_score = []
    # <-----------Training Loop---------------------------->
    pbar = tqdm(train_loader, desc="description")
    for x_train, y_train in pbar:
        x_train = torch.autograd.Variable(x_train).cuda()
        y_train = torch.autograd.Variable(y_train).cuda()
        optimizer.zero_grad()
        output = model(x_train)
        # Loss
        loss = criterion(output, y_train)
        losses_value = loss.item()
        # Score
        score = accuracy_metric(output, y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(losses_value)
        train_score.append(score.item())
        # train_score.append(score)
        pbar.set_description(
            "Epoch: {}, loss: {}, IoU: {}".format(epoch + 1, losses_value, score)
        )

    # <---------------Validation Loop---------------------->
    with torch.no_grad():
        for image, mask in val_loader:
            image = torch.autograd.Variable(image).cuda()
            mask = torch.autograd.Variable(mask).cuda()
            output = model(image)
            ## Compute Loss Value.
            loss = criterion(output, mask)
            losses_value = loss.item()
            ## Compute Accuracy Score
            score = accuracy_metric(output, mask)
            valid_loss.append(losses_value)
            valid_score.append(score.item())

    total_train_loss.append(np.mean(train_loss))
    total_train_score.append(np.mean(train_score))
    total_valid_loss.append(np.mean(valid_loss))
    total_valid_score.append(np.mean(valid_score))
    print(
        "\n###############Train Loss: {}, Train IOU: {}###############".format(
            total_train_loss[-1], total_train_score[-1]
        )
    )
    print(
        "###############Valid Loss: {}, Valid IOU: {}###############".format(
            total_valid_loss[-1], total_valid_score[-1]
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
    src.utils.utils.save_ckp(checkpoint, False, chkpoint_file, model_file)

    ## TODO: save the model if validation loss has decreased
    if total_valid_loss[-1] <= valid_loss_min:
        print(
            "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                valid_loss_min, total_valid_loss[-1]
            )
        )
        # save checkpoint as best model
        src.utils.utils.save_ckp(checkpoint, False, chkpoint_file, model_file)
        valid_loss_min = total_valid_loss[-1]