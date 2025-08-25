import warnings
import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import wandb
from copy import deepcopy

from gastric_dataloader import build_data_dict, GastricDataLoader, get_train_transform
import utils

# ===== CONFIG =====
warnings.filterwarnings('ignore')
wandb.init(project='Gland_SegNorm')
config = wandb.config

# Cambia este valor para seleccionar modo
# 'baseline1' = colon sin adaptar (solo evalúa en gástrico)
# 'baseline2' = fine-tuning supervisado en gástrico
# 'teacher_student' = semi-supervisado
mode = 'baseline2'

colon_weights_path = "./best_dice_colon.pth"

# Rutas a splits gastric
base_dir = "patches_20xR/labeled"
train_data = build_data_dict(os.path.join(base_dir, "train/images"), os.path.join(base_dir, "train/masks"))
val_data = build_data_dict(os.path.join(base_dir, "val/images"), os.path.join(base_dir, "val/masks"))
test_data = build_data_dict(os.path.join(base_dir, "test/images"), os.path.join(base_dir, "test/masks"))

patch_size = (config.patchsize, config.patchsize)
batch_size = config.batchsize
early_stop_patience = 5
epochs = config.epochs
augmentation_prob = config.augmentation_prob
lambda_consistency = config.lambda_consistency
ema_decay = config.ema_decay

# ===== DataLoaders =====
tr_transforms = get_train_transform(patch_size, prob=augmentation_prob)

if mode in ['baseline2', 'teacher_student']:
    train_dl_labeled = GastricDataLoader(
        train_data,
        batch_size=batch_size,
        patch_size=patch_size,
        num_threads_in_multithreaded=4,
        crop_status=True,
        crop_type="random",
        tr_transforms=tr_transforms,
    )

if mode == 'teacher_student':
    unlabeled_base = "patches_20xR/unlabeled"
    unlabeled_data = build_data_dict(
        os.path.join(unlabeled_base, "images"), os.path.join(unlabeled_base, "masks")
    )
    train_dl_unlabeled = GastricDataLoader(
        unlabeled_data,
        batch_size=batch_size,
        patch_size=patch_size,
        num_threads_in_multithreaded=4,
        crop_status=True,
        crop_type="random",
    )
    weak_transforms = get_train_transform(patch_size, prob=0.0)
    strong_transforms = tr_transforms

val_dl = GastricDataLoader(
    val_data,
    batch_size=batch_size,
    patch_size=patch_size,
    num_threads_in_multithreaded=1,
    crop_status=False,
    crop_type="random",
)

test_dl = GastricDataLoader(
    test_data,
    batch_size=batch_size,
    patch_size=patch_size,
    num_threads_in_multithreaded=1,
    crop_status=False,
    crop_type="random",
)

# ===== DEVICE =====
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ===== MODELO =====
base_model = smp.Unet(
    encoder_name=config.encoder_model,
    decoder_use_batchnorm=True,
    in_channels=3,
    classes=config.n_class,
).to(device)
base_model.load_state_dict(torch.load(colon_weights_path, map_location=device))

if mode == 'teacher_student':
    teacher = deepcopy(base_model)
    for param in teacher.parameters():
        param.requires_grad = False
    student = deepcopy(base_model)
    optimizer = eval(config.optimizer)(student.parameters(), lr=float(config.learning_rate))
else:
    model = base_model
    optimizer = eval(config.optimizer)(model.parameters(), lr=float(config.learning_rate))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=early_stop_patience, factor=0.1, mode='max'
)

dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
xent = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)


def custom_loss(pred, target):
    xent_l = xent(pred, target)
    dice_l = dice_loss(pred, target)
    loss = xent_l + dice_l
    return loss, xent_l, dice_l


# ===== LOOP TRAIN =====
def train_epoch(model_train, loader, optimizer):
    model_train.train()
    batch_loss, batch_dice_l = [], []
    loader_iter = iter(loader)
    for _ in tqdm(range(len(loader))):
        batch = next(loader_iter)
        imgs = utils.min_max_norm(batch['data'])
        segs = np.where(batch['seg'] > 0.0, 1.0, 0.0).astype('float32')
        imgs = torch.from_numpy(imgs).to(device)
        segs = torch.from_numpy(segs).to(device)
        pred = model_train(imgs)
        loss, _, dice_l = custom_loss(pred, segs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.detach())
        batch_dice_l.append(dice_l.detach())
    return (
        torch.mean(torch.as_tensor(batch_loss)).item(),
        1 - torch.mean(torch.as_tensor(batch_dice_l)).item(),
    )


def evaluate(loader, model_eval):
    model_eval.eval()
    batch_loss, batch_dice_l = [], []
    loader_iter = iter(loader)
    with torch.no_grad():
        for _ in tqdm(range(len(loader))):
            batch = next(loader_iter)
            imgs = utils.min_max_norm(batch['data'])
            segs = np.where(batch['seg'] > 0.0, 1.0, 0.0).astype('float32')
            imgs = torch.from_numpy(imgs).to(device)
            segs = torch.from_numpy(segs).to(device)
            pred = model_eval(imgs)
            loss, _, dice_l = custom_loss(pred, segs)
            batch_loss.append(loss)
            batch_dice_l.append(dice_l)
    return (
        torch.mean(torch.as_tensor(batch_loss)).item(),
        1 - torch.mean(torch.as_tensor(batch_dice_l)).item(),
    )


def train_epoch_teacher_student(student, teacher, optimizer):
    student.train()
    teacher.eval()
    batch_loss, batch_sup, batch_cons, batch_dice_l = [], [], [], []
    labeled_iter = iter(train_dl_labeled)
    unlabeled_iter = iter(train_dl_unlabeled)
    for _ in tqdm(range(len(train_dl_labeled))):
        batch_l = next(labeled_iter)
        imgs_l = utils.min_max_norm(batch_l['data'])
        segs_l = np.where(batch_l['seg'] > 0.0, 1.0, 0.0).astype('float32')
        imgs_l = torch.from_numpy(imgs_l).to(device)
        segs_l = torch.from_numpy(segs_l).to(device)
        pred_l = student(imgs_l)
        sup_loss, _, dice_l = custom_loss(pred_l, segs_l)

        batch_u = next(unlabeled_iter)
        raw_imgs = batch_u['data']
        dummy_seg = np.zeros((raw_imgs.shape[0], 1, raw_imgs.shape[2], raw_imgs.shape[3]), dtype=np.float32)
        weak = weak_transforms(data=raw_imgs.copy(), seg=dummy_seg.copy())['data']
        strong = strong_transforms(data=raw_imgs.copy(), seg=dummy_seg.copy())['data']
        weak = utils.min_max_norm(weak)
        strong = utils.min_max_norm(strong)
        weak = torch.from_numpy(weak).to(device)
        strong = torch.from_numpy(strong).to(device)
        with torch.no_grad():
            teacher_pred = teacher(weak)
        student_pred = student(strong)
        cons_loss = torch.mean((student_pred - teacher_pred) ** 2)

        loss = sup_loss + lambda_consistency * cons_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data.mul_(ema_decay).add_(s_param.data * (1.0 - ema_decay))

        batch_loss.append(loss.detach())
        batch_sup.append(sup_loss.detach())
        batch_cons.append(cons_loss.detach())
        batch_dice_l.append(dice_l.detach())

    return (
        torch.mean(torch.as_tensor(batch_loss)).item(),
        torch.mean(torch.as_tensor(batch_sup)).item(),
        torch.mean(torch.as_tensor(batch_cons)).item(),
        1 - torch.mean(torch.as_tensor(batch_dice_l)).item(),
    )


# ===== MAIN =====
def main():
    if mode == 'baseline1':
        print("Running Baseline 1: Colon model, no adaptation.")
        test_loss, test_dice = evaluate(test_dl, base_model)
        print(f"[Baseline 1] Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}")

    elif mode == 'baseline2':
        print("Running Baseline 2: Fine-tuning with labeled gastric train set.")
        best_val_dice = 0
        for e in range(1, epochs + 1):
            train_loss, train_dice = train_epoch(model, train_dl_labeled, optimizer)
            val_loss, val_dice = evaluate(val_dl, model)
            wandb.log(
                {
                    'train_loss': train_loss,
                    'train_dice': train_dice,
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                    'epoch': e,
                }
            )
            print(
                f"[Epoch {e}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
            )
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                filename = f"model_ep{e}_dice{val_dice:.3f}_lr{config.learning_rate}_bs{config.batchsize}_pz{config.patchsize}.pth"
                torch.save(model.state_dict(), filename)
                wandb.save(filename)
            scheduler.step(val_dice)
        print("Fine-tuning complete. Evaluating on test set...")
        model.load_state_dict(torch.load(filename, map_location=device))
        test_loss, test_dice = evaluate(test_dl, model)
        print(f"[Baseline 2] Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}")

    elif mode == 'teacher_student':
        print("Running Teacher–Student semi-supervised training.")
        best_val_dice = 0
        for e in range(1, epochs + 1):
            train_loss, sup_loss, cons_loss, train_dice = train_epoch_teacher_student(
                student, teacher, optimizer
            )
            val_loss, val_dice = evaluate(val_dl, student)
            wandb.log(
                {
                    'train_loss': train_loss,
                    'sup_loss': sup_loss,
                    'cons_loss': cons_loss,
                    'train_dice': train_dice,
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                    'epoch': e,
                }
            )
            print(
                f"[Epoch {e}] Train Loss: {train_loss:.4f} | Sup Loss: {sup_loss:.4f} | Cons Loss: {cons_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
            )
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                filename = f"ts_model_ep{e}_dice{val_dice:.3f}_lr{config.learning_rate}_bs{config.batchsize}_pz{config.patchsize}.pth"
                torch.save(student.state_dict(), filename)
                wandb.save(filename)
            scheduler.step(val_dice)
        print("Training complete. Evaluating on test set...")
        student.load_state_dict(torch.load(filename, map_location=device))
        test_loss, test_dice = evaluate(test_dl, student)
        print(f"[Teacher-Student] Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}")


if __name__ == '__main__':
    main()

