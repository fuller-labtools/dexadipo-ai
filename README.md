<p align="left">
  <span style="display:inline-block; vertical-align:middle; margin-right:12px;">
    <img src="https://github.com/user-attachments/assets/8a29b7e3-f6c1-4fdf-8d1e-05cc19d88cf4" alt="DEXAdipo logo" height="220">
  </span>
  <span style="display:inline-block; vertical-align:middle;">
    <img src="https://github.com/user-attachments/assets/1d114f74-ed93-426a-ad42-0f93ea963471" alt="ML diagram" height="200">
  </span>
</p>


## DEXAdipo — Model Overview

**DEXAdipo** predicts **subcutaneous (SUBQ)** and **visceral (VAT)** adipose tissue mass (g) from **single-energy digital radiography (DR)** images of mice. It couples robust preprocessing with two dedicated convolutional regression models to deliver fast, reproducible, non-invasive adipose depot quantification. **Try the online version:** https://fullerlabtools.shinyapps.io/dexadipo-ai/

---

### Why this model?
Quantifying adipose depots across ages, diets, sexes and strains is labor-intensive and error-prone. DEXAdipo was built to **generalize across biological and imaging variation**, providing **continuous (gram) estimates** of SUBQ and VAT directly from DR images so researchers can scale phenotyping in large cohorts.

---

### What the model expects (inputs)
- **Imaging modality:** **Single-energy (low-energy) DR**.
- **Recommended tube voltage:** **~35–45 kVp**.
- **Image formats:** DICOM / TIFF / PNG / JPG.
---

### What the model produces (outputs)
- Two continuous predictions in grams:
  - **SUBQ** (cervical, axillary, dorsolumbar, inguinal, gluteal pads)
  - **VAT** (mesenteric, retroperitoneal, gonadal, perirenal pads)

---

### Training data and ground truth
To encourage **broad generalization**, training combined:
- **Populations:** Multiple mouse strains (e.g., **B6.Cg-Lepob/J (ob/ob)**, **NZO/HlLtJ**)
- **Sex:** Male and female
- **Diet:** Chow and Western style high sugar/fat diets
- **Diversity:** Age and tube voltage variation
- **Labels:** **Dissected depot weights** for SUBQ and VAT (grams), measured post-mortem
- **Validation split:** **10%** held out

---

### Architecture (two models: SUBQ and VAT)
Each depot is predicted by its **own CNN-based regression network**:
- Residual CNN backbone (conv-BN-ReLU, max pooling, **residual blocks**, dropout)
- Dense regression head with **L2 regularization**
- **Optimizer:** Adam
- **Primary loss:** MSE
- **Metrics:** Includes **R²** for interpretability
- **Training aids:** Early stopping, learning-rate scheduling, checkpointing, and fine-tuning at reduced LR
- **Framework:** **TensorFlow/Keras**

---

### Performance (held-out validation)
- **SUBQ:** **R² = 0.955**  
- **VAT:** **R² = 0.945**

These values reflect performance on previously unseen animals.

---

### Intended use & limitations
- **Research use only**; not a clinical diagnostic device.
- Best performance is achieved with **single-energy, low-energy DR (~35–45 kVp)** and the following conventions:  
- **Orientation:** mouse faces right (nose right) with the body horizontally aligned.  
- **ROI:** center on the spinal midline; keep the **torso fully inside** the ROI (the head may be cropped).  
- Performance may degrade for extreme phenotypes or imaging protocols outside the training distribution; consider fine-tuning the model with additional examples.

---

### Getting started (quick)
- **Model inference:** Use any Python environment with TensorFlow/Keras, and code named DEXAdipo_inference.py. For fine-tuning use DEXAdipo_train.py
- **Or try the online version:** https://fullerlabtools.shinyapps.io/dexadipo-ai/

---

### 1. System requirements

### 1.1 Operating systems

DEXAdipo is implemented in Python and TensorFlow/Keras and should run on any 64-bit OS that supports these libraries, including:

- Linux
- macOS (Intel or Apple Silicon, via a compatible Python + TensorFlow 2.x installation)
- Windows 10/11 (native Python or WSL2)

### 1.2 Software dependencies

Core Python dependencies (in addition to the standard library):

- Python ≥ 3.8  
- `tensorflow` (TensorFlow 2.x)  
- `numpy`  
- `pandas`  
- `scikit-image`  
- `scikit-learn`  
- `matplotlib` (used for plotting augmented images in training)

You can either install these directly (see below) or run:

```bash
pip install -r requirements.txt
```
### 1.3 Versions tested
Tested on the following configuration:
- OS: Windows 11, Version 25H2 (OS Build 26200.7462)
- System type: 64-bit operating system, x64-based processor
- CPU: 13th Gen Intel(R) Core(TM) i9-13900KF @ 3.00 GHz
- RAM: 64.0 GB 
- GPU: NVIDIA GeForce RTX 4080
- Python: 3.9.18
- TensorFlow: 2.10.0 (GPU build)
- CUDA toolkit: 11.8 (nvcc 11.8.89)

### 1.4 Non-standard hardware
Training: An NVIDIA GPU (≥ 8 GB VRAM) is strongly recommended for full training as described in the manuscript and in DEXAdipo_train.py.

Inference / demo: CPU-only is sufficient for small batches of DR images; predicting on a handful of DR images runs in seconds on a standard 4-core desktop or laptop.

---

### 2. Installation guide
### 2.1 Instructions
Clone the repository

```bash
git clone https://github.com/fuller-labtools/dexadipo-ai.git
cd dexadipo-ai
```
Create and activate a fresh Python environment (recommended)
```bash
# Linux/macOS
python -m venv dexadipo_env
source dexadipo_env/bin/activate
# or, on Windows:
dexadipo_env\Scripts\activate
```
Install dependencies
```bash
pip install --upgrade pip
pip install tensorflow numpy pandas scikit-image scikit-learn matplotlib
# or simply:
# pip install -r requirements.txt
```
If you have a CUDA-capable GPU, install the appropriate GPU-enabled TensorFlow build.

### Pretrained model weights
- Download the pretrained models (SUBQ_model.h5 and VAT_model.h5) from the GitHub Releases page, e.g. under tag v1.0.0.
- Place them in a convenient directory (for example):
  - models/subq/SUBQ_model.h5
  - models/vat/VAT_model.h5
- Point MODEL_PATH in DEXAdipo_inference.py (and/or your own scripts) to the appropriate .h5 file.

If you prefer, you can also train your own models from scratch using DEXAdipo_train.py (see Section 4).

### 2.2 Typical install time
On a “normal” desktop or laptop with a reasonable internet connection, creating the environment and installing the Python dependencies typically takes ~10–20 minutes.

---

### 3. Demo
### 3.1 Demo via the Shiny app (small example dataset)
The Shiny app includes a built-in example dataset under the dropdown:
- Example Images: Lean mouse · High adiposity mouse
You can use this directly to demo the model:
- Go to: https://fullerlabtools.shinyapps.io/dexadipo-ai/

In the image selection panel, choose:
- “Lean mouse” (low adiposity example)
- “High adiposity mouse” (high adiposity example)
Draw a reasonable ROI around the torso as instructed.
Run predictions.
Typical prediction ranges observed based on ROI selection
High_adiposity_example
- SUBQ: ~5.53–5.96 g
- VAT: ~7.77–10.60 g
Low_adiposity_example (Lean mouse)
- SUBQ: ~1.79–2.03 g
- VAT: ~2.61–3.21 g 
These ranges reflect small variations due to ROI placement but are robust to reasonable differences.

Expected run time (demo via Shiny):
- Prediction for each example image is essentially instantaneous from the user’s perspective.

### 3.2 Local Python demo
If you prefer a local Python demo:
Prepare a small folder of DR images (e.g. DICOMs, TIFFs, PNGs) and a CSV with a FILENAMES column:
```bash
demo/
  demo_images/
    High_adiposity_example.png
    Low_adiposity_example.png
  demo_newdata.csv
```

Example demo_newdata.csv:
```bash
FILENAMES
High_adiposity_example.png
Low_adiposity_example.png
```

Edit DEXAdipo_inference.py:
```bash
MODEL_PATH = "path/to/your_trained_model.h5"
NEW_DATASET_CSV = "demo/demo_newdata.csv"
BASE_DIR = "demo/demo_images"
OUTPUT_CSV = "demo/demo_predictions.csv"
```
Run:
```bash
python DEXAdipo_inference.py
```

The script prints a table as follows:
```bash
Filename                     Predicted
High_adiposity_example.png   9.1
Low_adiposity_example.png    2.9
```

(Values will differ depending on which trained model you use and the actual images.)

The same results are saved to demo/demo_predictions.csv.

#### Expected run time (local demo):
For 2–10 images on a standard CPU-only desktop, inference completes in a few seconds (≪ 1 minute).

---

### 4. Instructions for use (running on your own data)
### 4.1 Preparing your data

Acquire single-energy DR images and export them to any of the supported formats.

Ensure:
- Mice are oriented consistently (nose to the right, body horizontal).
- The torso is fully visible in the ROI for the method you use.

For training, create a CSV file with at least:
- FILENAMES – image file names (e.g. mouse1.png)
- SUBQ – numeric labels for subcutaneous depot mass in grams

(For other depots, use the corresponding label column instead of SUBQ — for example VAT for visceral fat, or any other depot name you choose — and make sure the training script reads from that column.)

Store images and CSV in a directory structure such as:
```bash
data/
  train_images/
    mouse1.png
    mouse2.png
    ...
  train_labels_subq.csv
```

### 4.2 Training a new SUBQ model from scratch (DEXAdipo_train.py)

The training script is configured by default for SUBQ regression.

Open DEXAdipo_train.py and set the CONFIG paths:
```bash
LABEL_CSV      = "data/train_labels_subq.csv"   # CSV with FILENAMES and SUBQ
IMAGE_BASE_DIR = "data/train_images"            # Directory containing PNG images
MODELS_DIR     = "models/subq"                  # Output directory for models/checkpoints
```

The CSV must contain at least:
```bash
FILENAMES,SUBQ
mouse1.png,3.45
mouse2.png,5.12
...
```
Run the training script:
```bash
python DEXAdipo_train.py
```

#### What the script does:
- Loads and checks your label CSV (requires FILENAMES and SUBQ)
- Loads PNG images from IMAGE_BASE_DIR and preprocesses them
- Splits the data into:
  - Train
  - Validation
  - Held-out temp/test subset (X_temp, y_temp)
- Builds a residual CNN via regression_model()
- Compiles with:
  - loss = combined_loss (MSE + 1 − R²)
  - metrics = ["mse", RSquare()]

Uses data augmentation (ImageDataGenerator) with:
- rotation_range = 5
- width_shift_range = 0.1
- height_shift_range = 0.1

#### Phase 1:
Trains up to 8000 epochs with:
- ModelCheckpoint saving to best_initial_model.h5 (best val_loss)
- ReduceLROnPlateau (factor 0.7, patience 800, min LR 1e-6)
- A custom callback that logs validation MSE and R²

#### Phase 2:
Loads best_initial_model.h5, recompiles with a smaller LR (1e-6), and:
- Trains 1 epoch at a time for 800 epochs
- Saves each epoch as second_phase_model_epoch_{epoch}.h5
- Logs validation metrics via the custom callback

#### Model selection and final save:
After training, the script:
- Iterates over all second_phase_model_epoch_*.h5 models
- Evaluates each on the held-out X_temp, y_temp
- Computes MSE and R²
- Computes a combined score:
  - combined_score = 0.8 * R² + 0.6 * (1 / (1 + MSE))
- Selects the model with the best combined score
- Prints the best model path, its MSE and R² on the test data
- Saves the final selected model as:
  - MODELS_DIR/final_best_model.h5


#### Training time:
The script measures and prints:
- Phase 1 training time
- Phase 2 training time
- Total training time

On a modern GPU (e.g. RTX 4080), the full training procedure is on the order of ~1 hour (exact time depends on dataset size and hardware).

### 4.3 Running inference on new images (DEXAdipo_inference.py)

DEXAdipo_inference.py loads a trained model (either your own or the included pretrained SUBQ/VAT models) and applies it to new DR images.

#### Using the included SUBQ and VAT models on your own images
The repository includes pretrained models released under Releases → v1.0.0:
- SUBQ_model.h5 – predicts subcutaneous adipose tissue (SUBQ) mass
- VAT_model.h5 – predicts visceral adipose tissue (VAT) mass

To use these with your own images:

#### Prepare your DR images
- Acquire single-energy, low-energy DR images at ~35–45 kVp.
- Accepted formats: DICOM / TIFF / PNG / JPG.
- Use a DR acquisition pipeline where the background is approximately white and the mouse appears dark/black relative to the background.
- Set the global Pixels/mm resolution consistently across images, for example:
  - Faxitron® UltraFocusDXA: ~21.33 pixels/mm
  - MEDIKORS InAlyzer DEXA: ~3.90 pixels/mm

#### Standardise orientation and ROI
For best agreement with the pretrained models:
- Orient each mouse so the nose points to the right and the body is horizontal.
- Define a torso ROI so that:
  - The vertebral column is centred on the vertical midline of the ROI.
  - The cranial/upper border of the ROI transects the head just rostral to the pinnae/ear canals (i.e. include the caudal skull but exclude the snout/face).
  - The torso is fully inside the ROI.

If you are using the Shiny app to explore ROI placement, mimic that same ROI when exporting images for Python inference.

#### Organise files and create a CSV
Put your preprocessed DR images in a folder, e.g.:
```bash
data/
  new_images/
    mouseA.png
    mouseB.png
    ...
  new_images.csv
```
new_images.csv must contain at least a FILENAMES column:
```bash
FILENAMES
mouseA.png
mouseB.png
```
#### Configure DEXAdipo_inference.py

Set new_img_size to match the resolution used for the released models (typically:
- new_img_size = (512, 512)

- To predict SUBQ using the pretrained model:
```bash
MODEL_PATH      = "path/to/SUBQ_model.h5"
NEW_DATASET_CSV = "data/new_images.csv"
BASE_DIR        = "data/new_images"
OUTPUT_CSV      = "results/new_predictions_SUBQ.csv"
```

To predict VAT using the pretrained model:
```bash
MODEL_PATH      = "path/to/VAT_model.h5"
NEW_DATASET_CSV = "data/new_images.csv"
BASE_DIR        = "data/new_images"
OUTPUT_CSV      = "results/new_predictions_VAT.csv"
```

If you have trained your own model, simply point MODEL_PATH to your file instead (e.g. models/subq/final_best_model.h5 or models/vat/final_best_model.h5).

Run inference
```bash
python DEXAdipo_inference.py
```
#### What the script does
- Loads the model from MODEL_PATH with:
```bash
best_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"RSquare": RSquare, "combined_loss": combined_loss},
    compile=False
)
```
- Reads NEW_DATASET_CSV and checks for a FILENAMES column.
- For each filename:
  - Constructs img_path = os.path.join(BASE_DIR, FILENAMES)
  - Loads the image via imread
  - Resizes to new_img_size
  - Rescales intensities to [0, 1] (with a zero-range safeguard)
  - Adds a channel dimension to form (H, W, 1)
 - Stacks all images into a NumPy array new_data.
 - Creates a DataFrame with:
  - Filename, Predicted
- Prints all predictions and saves them to OUTPUT_CSV.

#### Expected runtime (inference)
- For tens of images on CPU, inference typically runs in seconds to a couple of minutes, depending on CPU speed and model size.
- On GPU, inference is effectively instantaneous for typical batch sizes.
