import os
import numpy as np
from tqdm import tqdm
import torchvision
import torchvision.transforms as tfms
import torch
import torch.utils.data as data
from src.data.loader import CustomDataLoader
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
)
from src.data.preprocessing import resize, normalize, torch_equalize
from src.model.unet import UNet
import src
from src.model.losses import DiceLoss
from src.model.metrics import IoU, Threshold_IoU
from sklearn.model_selection import train_test_split
from src.utils.utils import list_files

# splitting data into train and val sets
files = list_files(train_dir)
lables = list_files(labels_dir)
train_files, val_files, train_labels, val_labels = train_test_split(
    files, lables, train_size=train_val_splitting_ratio, random_state=seed
)


# Prepare Training Data Generator
train_dataset = CustomDataLoader(
    train_dir,
    labels_dir,
    train_files,
    train_labels,
    skip_blank=skip_empty,
    shuffle=True,
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
    ),
)


# Prepare Val Data Generator
val_dataset = CustomDataLoader(
    train_dir,
    labels_dir,
    val_files,
    val_labels,
    skip_blank=False,
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
    ),
)

train_loader = data.DataLoader(
    train_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=0,
)

val_loader = data.DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=batch_size,
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
valid_loss_min = np.Inf


total_train_loss = []
total_train_score = []
total_train_score_round = []
total_valid_loss = []
total_valid_score = []
total_valid_score_round = []


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

    # <-----------Training Loop---------------------------->
    # reset the counters
    train_dataset.reset_counters()
    val_dataset.reset_counters()
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
        score_t = threshold_metric(output, y_train)
        # Optimizing
        loss.backward()
        optimizer.step()
        # Logging
        train_loss.append(losses_value)
        train_score.append(score.item())
        train_score_round.append(score_t.item())
        pbar.set_description(
            "Epoch: {}, loss: {}, IoU: {}, t_IoU: {}".format(
                epoch + 1, losses_value, score, score_t
            )
        )

    # <---------------Validation Loop---------------------->
    model.eval()
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
            score_t = threshold_metric(output, mask)
            # logging
            valid_loss.append(losses_value)
            valid_score.append(score.item())
            valid_score_round.append((score_t.item()))

    total_train_loss.append(np.mean(train_loss))
    total_train_score.append(np.mean(train_score))
    total_valid_loss.append(np.mean(valid_loss))
    total_valid_score.append(np.mean(valid_score))
    total_train_score_round.append(np.mean(train_score_round))
    total_valid_score_round.append(np.mean(valid_score_round))
    print(
        "\n###########Train Loss: {}+-{}, Train IOU: {}+-{}, Train Threshold IoU: {}+-{}###########".format(
            total_train_loss[-1],
            np.std(train_loss),
            total_train_score[-1],
            np.std(train_score),
            total_train_score_round[-1],
            np.std(train_score_round),
        )
    )
    print(
        "###########Valid Loss: {}+-{}, Valid IOU: {}+-{}, Valid Threshold IoU: {}+-{}###########".format(
            total_valid_loss[-1],
            np.std(valid_loss),
            total_valid_score[-1],
            np.std(valid_score),
            total_valid_score_round[-1],
            np.std(valid_score_round),
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

    if total_valid_loss[-1] <= valid_loss_min:
        print(
            "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                valid_loss_min, total_valid_loss[-1]
            )
        )
        # save checkpoint as best model
        src.utils.utils.save_ckp(checkpoint, False, chkpoint_file, model_file)
        valid_loss_min = total_valid_loss[-1]

        # keeping track of current best model (for early stopping)
        best_current_checkpoint = checkpoint
        best_current_checkpoint_file = chkpoint_file
        best_current_model_file = model_file
        epochs_no_improve = 0

    else:
        # epoch passed without improvement
        epochs_no_improve += 1

    # checking for early stopping
    if epochs_no_improve > max_epochs_no_improve:
        # saving model as best model
        src.utils.utils.save_ckp(
            best_current_checkpoint,
            True,
            best_current_checkpoint_file,
            best_current_model_file,
        )
