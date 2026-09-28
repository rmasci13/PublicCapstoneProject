# Early Skin Cancer Detection via Explainable Deep Learning & LLM-Powered Interpretation

An end-to-end medical computer vision and explainable AI (XAI) application designed for multi-class skin lesion classification. By combining dual-branch neural networks (dermoscopic image + patient clinical metadata) with visual feature attribution maps (Grad-CAM and LIME) and LLM-synthesized narrative reports, this project aims to encourage skin self-exams and promote early clinical detection.

> **Disclaimer:** This application is designed solely for educational, research, and early awareness purposes. It is **not** a diagnostic medical tool and is **not** a substitute for professional clinical evaluation by a dermatologist.

## Technical Overview

The system evaluates dermoscopic images along with patient demographic metadata through a multi-stage machine learning and explainability pipeline:

1. **Multimodal Deep Learning Architecture:** Combines a convolutional neural network (CNN) for dermoscopic image feature extraction with an artificial neural network (ANN) for structured patient metadata (age, sex, anatomical site).
2. **Explainable AI (XAI) Visual Layer:** Applies dual heatmapping techniques, namely Grad-CAM (gradient-based) and LIME (perturbation-based), to highlight diagnostic visual regions for clinical interpretability and cross-verify feature attribution.
3. **LLM Narrative Synthesis:** Passes predictions, class confidence scores, and visual attention data to Google Gemini API to produce an accessible, plain-language assessment that emphasizes timely medical follow-up.

## Tech Stack & Tools

* **Core Language:** Python
* **Deep Learning & ML:** PyTorch, Torchvision, Scikit-Learn, NumPy, Pandas
* **Computer Vision & XAI:** OpenCV, Grad-CAM, LIME
* **Web Application:** Django (Backend REST API), Vue.js (Reactive Frontend)
* **Generative AI:** Google Gemini API

## Dataset

Primary dataset utilized is the **ISIC 2019 Challenge Dataset**:

* **Images:** Over 25,000 dermoscopic images across multiple diagnostic categories (Melanoma, Melanocytic Nevus, Basal Cell Carcinoma, etc.).
* **Metadata:** Patient age, biological sex, and anatomical lesion location.
* **Pre-processing:** Imbalance handling (class weighting/data augmentation), spatial normalization, standardizing crop and scale, and one-hot encoding categorical metadata.

## Methodology & Development Roadmap

* [x] **Phase 1: Exploratory Data Analysis & Preprocessing**
  * Dataset analysis and handling class imbalance (Melanocytic Nevus ~50%, Melanoma ~17.85%).
  * Preprocessing pipelines for images (flipping, rotations, normalization) and metadata normalization.
* [ ] **Phase 2: Single-Model Benchmarking & Transfer Learning**
  * Evaluating ImageNet pre-trained CNNs: `ResNet18`, `ResNet50`, `EfficientNet`, and `VGG16`.
  * Comparing pre-trained weights vs. training from scratch.
* [ ] **Phase 3: Multimodal Fusion & Metadata Tuning**
  * Building and tuning a standalone ANN for structured patient metadata.
  * Constructing dual-branch feature fusion model (Optimal CNN + ANN).
  * Cross-validation and hyperparameter optimization focused on PR-AUC, F1-scores, and minimizing false negatives.
* [ ] **Phase 4: Explainability & LLM Integration**
  * Generating dual Grad-CAM and LIME heatmaps.
  * Integrating Google Gemini API for prompt-engineered natural language summary output.
* [ ] **Phase 5: Full-Stack Web Application Deployment**
  * Building Django backend REST endpoints and Vue.js interactive user portal.
  * Implementing non-persistence user privacy guarantees (uploaded images are processed in-memory and not stored).

## Ethical Considerations & Privacy

* **Zero-Storage Privacy Policy:** User-submitted images and metadata are processed transiently in-memory and are never logged, stored, or re-transmitted.
* **Optimization for Recall:** Hyperparameter selection and loss function designs prioritize minimizing false negatives to prevent false reassurance regarding malignant lesions.
* **Framing & Disclaimers:** Every user-facing view includes explicit guidance emphasizing that output is purely informational and encouraging direct consultation with a qualified dermatologist.

## References

1. American Academy of Dermatology. *Skin Cancer Statistics*.
2. ISIC Challenge 2019. *International Skin Imaging Collaboration*.
3. He, K., et al. *Deep Residual Learning for Image Recognition*. CVPR 2016.
4. Selvaraju, R. R., et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV 2017.
5. Ribeiro, M. T., et al. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD 2016.
