import warnings
import os
import math
import time
import numpy as np
import torch
from torch import nn
import segmentation_models_pytorch as smp
from tqdm import tqdm
import wandb

from gastric_dataloader import build_data_dict, GastricDataLoader, get_train_transform
import utils  # tu módulo utils original
from copy import deepcopy
# ===== CONFIG =====
warnings.filterwarnings('ignore')
wandb.init(project='Gland_SegNorm')
config = wandb.config

# Cambia este valor para seleccionar modo
# 'baseline1' = colon sin adaptar (solo evalúa en gástrico)
# 'baseline2' = fine-tuning supervisado en gástrico
mode = 'baseline2'

colon_weights_path = "./best_dice_colon.pth"

# Rutas a splits gastric
base_dir = "patches_20xR/labeled"
train_data = build_data_dict(os.path.join(base_dir, "train/images"), os.path.join(base_dir, "train/masks"))
val_data   = build_data_dict(os.path.join(base_dir, "val/images"), os.path.join(base_dir, "val/masks"))
test_data  = build_data_dict(os.path.join(base_dir, "test/images"), os.path.join(base_dir, "test/masks"))

patch_size = (config.patchsize,config.patchsize)
batch_size = config.batchsize
early_stop_patience = 5
epochs = config.epochs
augmentation_prob = config.augmentation_prob
# ===== DataLoaders =====
tr_transforms = get_train_transform(patch_size, prob=augmentation_prob) # data augmentation

if mode == 'baseline2':
    train_dl = GastricDataLoader(train_data, batch_size=batch_size, patch_size=patch_size,
                                 num_threads_in_multithreaded=4, crop_status=True, crop_type="random",tr_transforms=tr_transforms)
    train_gen = iter(train_dl)

val_dl = GastricDataLoader(val_data, batch_size=batch_size, patch_size=patch_size,
                           num_threads_in_multithreaded=1, crop_status=True,       # ¡IMPORTANTE!
    crop_type="random" )
val_gen = iter(val_dl)

test_dl = GastricDataLoader(test_data, batch_size=batch_size, patch_size=patch_size,
                            num_threads_in_multithreaded=1, crop_status=True,       # ¡IMPORTANTE!
    crop_type="random" )
test_gen = iter(test_dl)

# ===== DEVICE =====
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ===== MODELO =====
model = smp.Unet(encoder_name=config.encoder_model, decoder_use_batchnorm=True,
                 in_channels=3, classes=config.n_class).to(device)
model.load_state_dict(torch.load(colon_weights_path, map_location=device))

optimizer = eval(config.optimizer)(model.parameters(), lr=float(config.learning_rate))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=early_stop_patience, factor=0.1,mode='max')

dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
xent = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)

def custom_loss(pred, target):
    xent_l = xent(pred, target)
    dice_l = dice_loss(pred, target)
    loss = xent_l + dice_l
    return loss, xent_l, dice_l

# ===== LOOP TRAIN =====
def train_epoch(model, optimizer):
    model.train()
    batch_loss, batch_dice_l = [], []
    for _ in tqdm(range(len(train_dl) // batch_size)):
        batch = next(train_gen)
        imgs = utils.min_max_norm(batch['data'])
        segs = np.where(batch['seg'] > 0., 1.0, 0.).astype('float32')
        imgs, segs = torch.from_numpy(imgs).to(device), torch.from_numpy(segs).to(device)
        pred = model(imgs)
        loss, _, dice_l = custom_loss(pred, segs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss)
        batch_dice_l.append(dice_l)
    return torch.mean(torch.as_tensor(batch_loss)).item(), 1 - torch.mean(torch.as_tensor(batch_dice_l)).item()

def evaluate(loader):
    model.eval()
    batch_loss, batch_dice_l = [], []
    with torch.no_grad():
        for _ in tqdm(range(len(loader) // batch_size)):
            batch = next(loader)
            imgs = utils.min_max_norm(batch['data'])
            segs = np.where(batch['seg'] > 0., 1.0, 0.).astype('float32')
            imgs, segs = torch.from_numpy(imgs).to(device), torch.from_numpy(segs).to(device)
            pred = model(imgs)
            loss, _, dice_l = custom_loss(pred, segs)
            batch_loss.append(loss)
            batch_dice_l.append(dice_l)
    return torch.mean(torch.as_tensor(batch_loss)).item(), 1 - torch.mean(torch.as_tensor(batch_dice_l)).item()

# ===== MAIN =====
def main():
    if mode == 'baseline1':
        print("Running Baseline 1: Colon model, no adaptation.")
        test_loss, test_dice = evaluate(test_gen)
        print(f"[Baseline 1] Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}")

    elif mode == 'baseline2':
        print("Running Baseline 2: Fine-tuning with labeled gastric train set.")
        best_val_dice = 0
        best_epoch = 0

        for e in range(1, epochs + 1):
            train_loss, train_dice = train_epoch(model, optimizer)
            val_loss, val_dice = evaluate(val_gen)

            print(f"[Epoch {e}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_model_wts = deepcopy(model.state_dict())
                best_epoch = e
                filename = f"model_ep{e}_dice{val_dice:.3f}_lr{config.learning_rate}_bs{config.batchsize}_pz{config.patchsize}.pth"
                torch.save(model.state_dict(), filename)
                wandb.save(filename)

            scheduler.step(val_dice)
        print("Fine-tuning complete. Evaluating on test set...")
        model.load_state_dict(torch.load(filename, map_location=device))
        test_loss, test_dice = evaluate(test_gen)
        print(f"[Baseline 2] Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}")


if __name__ == '__main__':
    main()
