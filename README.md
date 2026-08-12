# OVANCO — Ovarian Cancer Detection Platform

> AI-powered ovarian cancer screening from ultrasound scans — built for patients and clinicians.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)
![Accuracy](https://img.shields.io/badge/Accuracy-76.68%25-green)
![AUC](https://img.shields.io/badge/AUC-0.91-brightgreen)

---

## What is OVANCO?

OVANCO is a web-based AI screening tool that analyses ovarian ultrasound images to detect the possible presence of ovarian cancer. Upload a scan and receive:

- A **3-class classification** — Normal / Benign / Malignant
- A **GradCAM heatmap** highlighting regions that influenced the prediction
- A **confidence score** and risk level (Low / Moderate / High)
- A **probability breakdown** across all three classes

---

## Motivation

Ovarian cancer has one of the highest mortality rates among gynaecological cancers — largely because it is diagnosed late. Ultrasound imaging is the most accessible screening tool but accurate interpretation requires specialist expertise not always available in resource-limited settings. OVANCO bridges that gap as an AI-assisted first-pass screening tool.

---

## Model Performance

| Metric | Value |
|---|---|
| Architecture | EfficientNetB3 (Transfer Learning) |
| Dataset | MMOTU OTU_2D — 1,469 clinically annotated images |
| Test Accuracy | 76.68% |
| Test AUC | 0.905 |
| Classes | Normal / Benign / Malignant |
| Visualization | GradCAM (Gradient-weighted Class Activation Mapping) |

### Per-class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 64% | 71% | 0.67 |
| Benign | 83% | 81% | 0.82 |
| Malignant | 72% | 71% | 0.72 |

---

## Dataset

Trained on the **MMOTU (Multi-Modality Ovarian Tumor Ultrasound)** dataset:
- 1,469 clinically annotated 2D ultrasound images
- Collected from Beijing Shijitan Hospital, Capital Medical University
- Annotated by 27 gynaecology clinical experts
- Published: Zhao et al., IEEE Transactions 2022
- 8 tumour categories remapped to 3 classes (Normal / Benign / Malignant)
- Train/Val/Test split: 70% / 15% / 15%

---

## Tech Stack

- **Backend:** Python, Flask
- **Deep Learning:** TensorFlow, Keras, EfficientNetB3
- **Image Processing:** OpenCV, Pillow, NumPy
- **Visualization:** GradCAM via GradientTape
- **Deployment:** Render (Gunicorn)

---

## Project Structure

```
OVANCO/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Procfile                  # Render/Heroku deployment
├── render.yaml               # Render configuration
├── DEPLOY.md                 # Deployment guide
├── train_model.py            # Training script
├── MMOTU_to_3class.py        # Dataset remapping script
├── .gitignore
├── models/
│   └── README.md             # How to get model files
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── upload.html
│   ├── results.html
│   ├── about.html
│   ├── 404.html
│   └── 500.html
├── static/
│   └── css/
│       └── style.css
└── uploads/                  # Auto-created at runtime
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/OVANCO.git
cd OVANCO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add model file
Download `ovarian_cancer_model_v2.h5` from the link below and place in `models/`:
```
models/ovarian_cancer_model_v2.h5
```
Model download: [Google Drive Link](YOUR_DRIVE_LINK_HERE)

### 4. Run locally
```bash
python app.py
```
Visit `http://localhost:5000`

### 5. Demo login
```
Doctor:  doctor@ovanco.com  /  doctor123
Patient: patient@ovanco.com /  patient123
```

---

## How GradCAM Works

GradCAM (Gradient-weighted Class Activation Mapping) computes the gradient of the predicted class score with respect to the input image. Regions with higher gradient magnitude contributed more to the prediction:

- **Red/Orange** = High model attention — regions that most influenced the prediction
- **Blue** = Low model attention — background regions

Unlike a separate saliency model, GradCAM runs directly on the classification model — making it more accurate, interpretable, and consistent with the prediction.

---

## Medical Disclaimer

OVANCO is an AI-assisted screening tool built for **educational and research purposes only**.

It is **not approved for clinical use** and must not be used as the sole basis for any medical decision. Always consult a qualified gynaecologist or oncologist for diagnosis and treatment.

---

## Author

**Aryaa Deshmukh**
B.Tech Bioengineering (Bioinformatics & Data Analytics)

---

## Citation

If you use OVANCO or the MMOTU dataset in your work:

```
Zhao, Y. et al. (2022). MMOTU: A Multi-Modality Ovarian Tumor Ultrasound 
Image Dataset for Unsupervised Cross-Domain Semantic Segmentation. 
IEEE Transactions. Beijing Shijitan Hospital, Capital Medical University.
```
