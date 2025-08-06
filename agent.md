# Agent Instructions for GastricGlandSegmentation

## Project Context
This repository contains experiments, baselines, and semi-supervised learning (Teacher–Student) pipelines for gastric gland segmentation, adapted from a colon-trained model (GlaS dataset) to gastric histopathology images.

## Main Goals
- Maintain clear, modular, and reproducible code for segmentation experiments.
- Allow switching between Baseline 1 (no adaptation), Baseline 2 (fine-tuning), and Teacher–Student from a single entry script via a `mode` flag.
- Ensure all training, validation, and test splits are **case-based**, not patch-based, to avoid data leakage.

## Coding Guidelines
1. **Structure**
   - Keep data loading code in `gastric_dataloader.py` or equivalent.
   - Store augmentation/transforms in a separate module if they grow complex.
   - Keep model definitions in `models/` if they differ from default SMP models.
   - Store training loops in a main experiment script (`experiments.py`).

2. **Data**
   - Input structure:
     ```
     patches_20xR/
         labeled/
             train/images/
             train/masks/
             val/images/
             val/masks/
             test/images/
             test/masks/
         unlabeled/  # For Teacher–Student
     ```
   - Use `crop_type="random"` for training to avoid positional bias.
   - Use deterministic seeds for reproducibility (`random.seed`, `np.random.seed`, `torch.manual_seed`).

3. **Experiments**
   - Baseline 1: Load colon weights → evaluate only on gastric test set.
   - Baseline 2: Load colon weights → fine-tune on gastric train → evaluate on test.
   - Teacher–Student: Train student with labeled (supervised) + unlabeled (consistency loss).

4. **Logging**
   - Use Weights & Biases (`wandb`) for tracking metrics and visualizations.
   - Save best models by both **loss** and **Dice score**.

5. **Code Style**
   - Follow PEP8.
   - Use clear variable names: `train_loss`, `val_dice`, `test_dl`, etc.
   - Avoid hard-coded paths; use config files or command-line arguments.

## Agent Role
When editing or generating code for this repository:
- **Preserve the directory structure** and file separation.
- Ensure all generated code is compatible with PyTorch, SMP, and current data loader.
- Minimize breaking changes: keep interfaces similar so experiments remain runnable.
- Automatically integrate new features (e.g., Teacher–Student) into the main loop with minimal duplication.
