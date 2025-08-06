import os
import shutil
import random
import cv2
import numpy as np
from collections import defaultdict

# Paths
base_dir = "./patches_20xR/labeled"
images_dir = os.path.join(base_dir, "images")
masks_dir = os.path.join(base_dir, "masks")

# Nuevo destino
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, "masks"), exist_ok=True)

# Obtener case_ids (antes de "_patch" o "_neg")
def get_case_id(filename):
    return filename.split("_patch")[0].split("_neg")[0]

case_to_files = defaultdict(list)
for fname in os.listdir(images_dir):
    if fname.endswith(".png"):
        cid = get_case_id(fname)
        case_to_files[cid].append(fname)

# Lista de casos y shuffle
all_cases = list(case_to_files.keys())
random.seed(42)
random.shuffle(all_cases)

# División 32 / 8 / 16 casos
train_cases = all_cases[:32]
val_cases = all_cases[32:40]
test_cases = all_cases[40:]

print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")

# Función para contar glándulas en una máscara
def count_glands(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 127).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_bin)
    return max(0, num_labels - 1)  # restar 1 para excluir el fondo

# Copiar archivos y contar glándulas
def copy_and_count(cases, split_name):
    total_glands = 0
    total_patches = 0
    for cid in cases:
        for fname in case_to_files[cid]:
            # Copiar imagen y máscara
            shutil.copy(os.path.join(images_dir, fname),
                        os.path.join(base_dir, split_name, "images", fname))
            shutil.copy(os.path.join(masks_dir, fname),
                        os.path.join(base_dir, split_name, "masks", fname))
            # Contar glándulas
            mask_path = os.path.join(masks_dir, fname)
            glands_in_patch = count_glands(mask_path)
            total_glands += glands_in_patch
            total_patches += 1
    print(f"{split_name}: {total_patches} parches, {total_glands} glándulas")
    return total_glands

glands_train = copy_and_count(train_cases, "train")
glands_val = copy_and_count(val_cases, "val")
glands_test = copy_and_count(test_cases, "test")

print(f"\nResumen total:")
print(f"Train: {glands_train} glándulas")
print(f"Val: {glands_val} glándulas")
print(f"Test: {glands_test} glándulas")
