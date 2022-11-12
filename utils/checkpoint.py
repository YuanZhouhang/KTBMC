# author: Yuan Zhouhang
#
# ------------------------------------------------------------

import torch
import os
from timm.models.layers import trunc_normal_


def load_checkpoint(path):
    expid = input("Please input the last experiment ID:")
    checkpoint = torch.load(f"{path}/checkpoint_{expid}.pth")
    return checkpoint


def save_checkpoint(expid, model, optmz, epoch, loss, path):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer_state_dict': optmz.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, f"{path}/checkpoint_{expid}.pth")

