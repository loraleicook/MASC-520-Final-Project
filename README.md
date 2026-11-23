# MASC 520 Final Project: Using Deep Neural Networks to Predict Material Hardness

**Authors:** Jingchen Wu, Loralei Cook, and Carolina Quinonez

**Repository:** https://github.com/loraleicook/MASC-520-Final-Project

## Overview

This project predicts the hardness of a material from scanning electron microscope (SEM) images of its microstructure. It compares two simple baseline models built on hand-crafted image features against a convolutional neural network (CNN) that learns its own features from image patches.

| Model | Input | Idea |
|---|---|---|
| Ridge regression | 20 hand-crafted image features + magnification | Linear model with an L2 penalty |
| Random forest | 20 hand-crafted image features + magnification | Average of 200 decision trees |
| CNN (ResNet18 by default) | Random image patches + magnification | Transfer learning from ImageNet, fine-tuned on SEM images |

All models are evaluated with the same metrics (MAE, RMSE, R²) on a held-out test set.

## Repository Contents

```
.
├── MASC_520_Final_Project.ipynb   # Main notebook (data prep, baselines, CNN, results)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── data/                          # Not tracked by default (see "Data" below)
    ├── labels.csv
    └── images/
```

## Requirements

Python 3 with the packages below. Install them with:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
pandas
opencv-python
matplotlib
torch
torchvision
scikit-image
scikit-learn
```

Note that some import names differ from the package names: `cv2` is installed as `opencv-python`, `skimage` as `scikit-image`, and `sklearn` as `scikit-learn`.

A GPU is optional. The notebook detects one automatically and uses larger patches and more epochs when it finds it.

| Setting | GPU | CPU |
|---|---|---|
| Patch size | 224 px | 160 px |
| Epochs | 20 | 10 |

## Data

The notebook expects:

1. **`labels.csv`** with one row per image. The required columns are:
   - `filename`: the image file name 
   - `hardness`: the measured hardness 
2. **Optional columns:**
   - `magnification`: e.g. `60` or `500x`. If missing, it is parsed from the filename (e.g. `sample3_500x.png`).
   - `split`: values containing `test`, `hold` or `val` mark test images, and everything else is training data.
   - `sample_id`: identifies which physical sample an image came from. Used for grouped cross-validation so images from the same sample never appear in both the training and validation folds.
3. **Images** as individual files or inside a `.zip` (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`).


## How to Run

### In Google Colab (recommended)

1. Open `MASC_520_Final_Project.ipynb` in Colab.
2. Run the first (configuration) cell.
3. Run the data preparation cell and upload `labels.csv` and the images in a zip file when prompted. To load from Google Drive instead, set `USE_DRIVE = True` and update `DRIVE_CSV` and `DRIVE_ZIP`.
4. Run the remaining cells in order.

### Locally

1. Install the requirements.
2. Place `labels.csv` in `data/` and your images in `data/images/`.
3. Open the notebook in Jupyter and run all cells. The Colab upload step will print a message and skip itself.

## Method

### Data preparation
- Images are loaded as grayscale, optionally cropped at the bottom (`CROP_BOTTOM`) to remove the SEM information banner, and downscaled by `SCALE` (default 0.5).
- Magnification is one-hot encoded and passed to every model as an extra input.
- Cross-validation folds are built with `GroupKFold` on `sample_id` to avoid data leakage.

### Baselines: Ridge and Random Forest
For each image, 20 features are computed:
- Brightness statistics: mean, standard deviation, 5th and 95th percentiles
- Gray-level co-occurrence matrix (GLCM) texture: contrast, homogeneity, energy, correlation
- Local binary pattern (LBP) histogram: 10 bins
- Edge density (Canny) and sharpness (variance of the Laplacian)

Ridge regression (with automatically selected regularization strength) and a random forest of 200 trees are trained on these features. Cross-validation MAE and R² are reported on the training set.

### CNN regression
- A pretrained ResNet18 (or ResNet50 / EfficientNet-B0) has its classification layer removed and replaced with a small regression head (dropout and a linear layer). The head also receives the magnification features.
- **Training** uses random square patches with augmentation (flips, 90° rotations, brightness and contrast jitter). Each image is sampled `REPS` times per epoch.
- **Transfer learning schedule:** the pretrained body is frozen for the first `FREEZE_EPOCHS` (3) epochs so only the new head trains. After that, the whole network is fine-tuned with a lower learning rate for the body (1e-4) than for the head (1e-3).
- **Optimization:** AdamW, cosine learning-rate decay, and Smooth L1 (Huber) loss on standardized hardness values.
- **Prediction:** each test image is split into a 3x3 grid of 9 patches, the predictions are averaged, and the result is converted back to real hardness units.

## Configuration

Key settings in the first cell:

| Variable | Default | Meaning |
|---|---|---|
| `BACKBONE` | `"resnet18"` | `"resnet18"`, `"resnet50"` or EfficientNet-B0 |
| `PATCH` | 224 (GPU) / 160 (CPU) | Patch size in pixels |
| `EPOCHS` | 20 (GPU) / 10 (CPU) | Training epochs |
| `REPS` | 4 | Random patches per image per epoch |
| `FREEZE_EPOCHS` | 3 | Epochs with the pretrained body frozen |
| `BATCH` | 16 | Batch size |
| `SEED` | 42 | Random seed |
| `RUN_CV` | `False` | Set to `True` to cross-validate the CNN (slow: trains 5 models) |
| `CROP_BOTTOM` | 0 | Pixel rows to remove from the bottom of each image |
| `TAB_COLS` | `[]` | Extra numeric columns (e.g. grain size) to use as inputs |

## Outputs

- A table of **MAE, RMSE and R²** on the test set for each model
- A per-image table of measured vs. predicted hardness
- A three-panel figure: CNN training loss, a **parity plot** (measured vs. predicted), and CNN residuals per test image
- **`hardness_cnn.pt`**: saved CNN weights, with the backbone name and the hardness mean and standard deviation used for standardization

## Evaluation Metrics

- **MAE** (mean absolute error): average size of the prediction error, in hardness units
- **RMSE** (root mean squared error): like MAE but penalizes large errors more
- **R²**: fraction of the variation in hardness explained by the model (1 is perfect, 0 is no better than predicting the average)

## Notes and Limitations

- The test set is small, so test metrics can be noisy. Use the cross-validation results as a cross-check.
- The saved checkpoint does not include the scaler used for the tabular features. It must be rebuilt to reuse the model for new predictions.
- If the CSV has no `sample_id` column, each image is treated as its own group, and images from the same sample can end up in both training and validation folds.
