import os
import torch
import shutil


def list_files(directory):
    """
    Helper function, which only makes a list of dir of none-hidden files:
    Params:
        directory: is the filepath
    Returns:
        a list of the files in the directory that don't start with '.'
    """
    return [f for f in os.listdir(directory) if not f.startswith(".")]


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    Saving a certain checkpoint/model state
    Params:
        state: checkpoint we want to save
        is_best: is this the best checkpoint; min validation loss
        checkpoint_path: path to save checkpoint
        best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    Loading a specific checkpoint/model state
    Params:
        checkpoint_fpath: path to saved checkpoint
        model: model that we want to load checkpoint parameters into
        optimizer: optimizer we defined in previous training
    Returns: loaded model, loaded optimizer, epoch (int) of loaded checkpoint, minimal val loss of retrieved model
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint["state_dict"])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint["optimizer"])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint["valid_loss_min"]
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint["epoch"], valid_loss_min.item()
