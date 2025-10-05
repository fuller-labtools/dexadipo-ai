<p align="left">
  <img src="https://github.com/user-attachments/assets/8a29b7e3-f6c1-4fdf-8d1e-05cc19d88cf4" alt="DEXAdipo logo" height="230" />
  &nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/5f338ce8-33e7-4174-89ac-cc61def4f22e" alt="ML diagram" height="220" />
</p>

## DEXAdipo — Model Overview

**DEXAdipo** predicts **subcutaneous (SUBQ)** and **visceral (VAT)** adipose tissue mass (g) from **single-energy digital radiography (DR)** images of mice. It couples robust preprocessing with two dedicated convolutional regression models to deliver fast, reproducible adipose quantification without manual pad dissection. **Try the online version:** https://fullerlabtools.shinyapps.io/dexadipo-ai/

---

### Why this model?
Quantifying adipose depots across ages, diets, sexes and strains is labor-intensive and error-prone. DEXAdipo was built to **generalize across biological and imaging variation**, providing **continuous (gram) estimates** of SUBQ and VAT directly from DR images so researchers can scale phenotyping in large cohorts.

---

### What the model expects (inputs)
- **Imaging modality:** **Single-energy (low-energy) DR**, not dual-energy DEXA.
- **Recommended tube voltage:** **~35–45 kVp** (empirically robust range).
- **Image formats:** DICOM / TIFF / PNG / JPG.
- **Resolution:** Works at arbitrary resolution; default is **21.3292 px/mm** (calibrated on **UltraFocusDXA by Faxitron®**).  
- **Pose/orientation:** Mouse **facing right** (nose to the right), body roughly horizontal.
  
---

### What the model produces (outputs)
- Two continuous predictions in grams:
  - **SUBQ** (cervical, axillary, dorsolumbar, inguinal, gluteal pads)
  - **VAT** (mesenteric, retroperitoneal, gonadal, perirenal pads)

---

### Training data and ground truth
To encourage **broad generalization**, training combined:
- **Populations:** Outbred + inbred (e.g., **B6.Cg-Lepob/J (ob/ob)**, **NZO/HlLtJ**)
- **Sex:** Male and female
- **Diet:** Chow and Western diet
- **Diversity:** Age and tube voltage variation
- **Labels:** **Dissected depot weights** for SUBQ and VAT (grams), measured post-mortem
- **Validation split:** **13%** held out (never seen during training)

---

### Preprocessing (for training and inference)
- **Auto inversion** to ensure white background, dark mouse (robust to TIFF/DICOM conventions)
- **Scaling** to a common density (default **21.3292 px/mm**)
- **Fixed ROI**: **2240×1600 px** cropped around the torso
- **Canvas**: ROI pasted onto **2300×2300 px** white canvas
- **Model input**: Normalized grayscale resized to **512×512**

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

These values reflect performance on previously unseen animals drawn from the same diverse distribution described above.

---

### Intended use & limitations
- **Research use only**; not a clinical diagnostic device.
- Best performance is achieved when images are **low-energy DR** within the **~35–45 kVp** range and follow the **orientation/ROI** conventions above.
- Extreme phenotypes or imaging protocols far outside training distribution may reduce accuracy; consider fine-tuning the model with additonal examples.

---

### Getting started (quick)
- **Model inference:** Use any Python environment with TensorFlow/Keras; normalize grayscale inputs as described in *Preprocessing* and pass through the SUBQ and VAT models for gram-level predictions.
- **Or try the online version:** https://fullerlabtools.shinyapps.io/dexadipo-ai/
---

### Citation
If you use DEXAdipo AI in a publication, please cite this repository and paper.
