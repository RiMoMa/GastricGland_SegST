# GastricGland_SegST

Repositorio para la segmentación de glándulas gástricas mediante redes neuronales convolucionales. Incluye scripts para extraer parches anotados, dividir el conjunto de datos por casos y ejecutar experimentos de adaptación desde un modelo entrenado en colon.

## Requisitos
- Python 3.11
- PyTorch y `segmentation-models-pytorch`
- `batchgenerators` y dependencias de procesamiento de imágenes (`opencv`, `shapely`, `openslide`)
- Se recomienda crear el entorno con `conda env create -f enviromentUnet.yml`

## Flujo de trabajo
1. **Extracción de parches**
   ```bash
   python 1_data_extraction.py
   ```
   Lee las imágenes WSIs y anotaciones XML para generar parches de 512×512 con sus máscaras utilizando `openslide` y geometría de polígonos.
2. **División del dataset**
   ```bash
   python 2_divide_dataset.py
   ```
   Separa las imágenes en particiones de entrenamiento, validación y prueba de forma *case-based* para evitar fuga de datos.
3. **Entrenamiento y evaluación**
   Ajusta la variable `mode` en `3_experiments.py` a `baseline1` (solo evaluación) o `baseline2` (fine-tuning) y ejecuta:
   ```bash
   python 3_experiments.py
   ```
   El script usa `segmentation_models_pytorch.Unet` con un encoder tipo ResNet, combina las pérdidas Dice y BCE, aplica aumentos de datos (deformaciones elásticas, rotaciones, reflejos, ruido) y registra métricas en Weights & Biases.

## Configuración
- Los hiperparámetros se definen mediante `wandb.config` y pueden explorarse con barridos (`config/sweep.yaml`).
- Los pesos pre-entrenados del modelo de colon se cargan desde `best_dice_colon.pth`.

## Estructura de datos esperada
```
patches_20xR/
    labeled/
        train/images, train/masks
        val/images, val/masks
        test/images, test/masks
```

## Licencia
Este repositorio se distribuye para fines de investigación académica.
