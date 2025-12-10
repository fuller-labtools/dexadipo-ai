<p align="left">
  <span style="display:inline-block; vertical-align:middle; margin-right:12px;">
    <img src="https://github.com/user-attachments/assets/8a29b7e3-f6c1-4fdf-8d1e-05cc19d88cf4" alt="DEXAdipo logo" height="220">
  </span>
  <span style="display:inline-block; vertical-align:middle;">
    <img src="https://github.com/user-attachments/assets/1d114f74-ed93-426a-ad42-0f93ea963471" alt="ML diagram" height="200">
  </span>
</p>


## DEXAdipo — Model Overview

**DEXAdipo** predicts **subcutaneous (SUBQ)** and **visceral (VAT)** adipose tissue mass (g) from **single-energy digital radiography (DR)** images of mice. It couples robust preprocessing with two dedicated convolutional regression models to deliver fast, reproducible non-invasive adipose depot quantification. **Try the online version:** https://fullerlabtools.shinyapps.io/dexadipo-ai/

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

### Citation
If you use DEXAdipo in a publication, please cite this repository.
