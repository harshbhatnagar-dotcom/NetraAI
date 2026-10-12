# Explainable AI for Diabetic Retinopathy Screening in Rural India

An experimental hackathon project combining **EfficientNetB0**, **Grad-CAM**, **Streamlit**, and **Groq-hosted Llama through the OpenAI Python SDK** to demonstrate diabetic retinopathy (DR) severity classification and generate a patient-friendly explanation.

> **Medical safety notice:** This is an unvalidated research prototype, not a medical device. Do not use it to diagnose disease, rule out diabetic retinopathy, choose treatment, or delay care. Results require assessment by a qualified eye-care professional.

## Contents

- [Project overview](#project-overview)
- [Features](#features)
- [DR classes](#dr-classes)
- [Dataset and preparation](#dataset-and-preparation)
- [Model training and evaluation](#model-training-and-evaluation)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the Streamlit app](#run-the-streamlit-app)
- [How the app works](#how-the-app-works)
- [Grad-CAM interpretation](#grad-cam-interpretation)
- [GenAI report](#genai-report)
- [Known limitations](#known-limitations)
- [Future work](#future-work)

## Project overview

Diabetic retinopathy is an eye complication associated with diabetes. This project explores an explainable-AI workflow that:

1. Accepts a retinal fundus image.
2. Preprocesses it to the model's expected input size.
3. Predicts one of five DR severity categories with a trained EfficientNetB0 classifier.
4. Generates a Grad-CAM visualization of regions that influenced the selected prediction.
5. Optionally asks a Groq-hosted language model to turn the model output into a concise report in English or Hinglish.

The predicted category and scores are **not a confirmed diagnosis**. Grad-CAM shows model influence, not verified lesions or disease locations.

## Features

- Upload a retinal fundus image (`.jpg`, `.jpeg`, or `.png`).
- Five-class DR prediction.
- Display model scores/probabilities for the five classes.
- Grad-CAM heatmap and overlay.
- Optional Groq GenAI explanation in the selected language.
- Patient-friendly next-step suggestions and a medical disclaimer.
- A single-file Streamlit implementation (`app.py`).

## DR classes

| Label | Class |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Dataset and preparation

The project used the **APTOS 2019 Blindness Detection** dataset. The downloaded archive was reported to be approximately 9.51 GB. The labeled training CSV contained 3,662 records, and all 3,662 corresponding labeled images were extracted and preprocessed successfully.

Reported class counts:

| Class label | Number of images |
|---:|---:|
| 0 — No DR | 1,805 |
| 1 — Mild | 370 |
| 2 — Moderate | 999 |
| 3 — Severe | 193 |
| 4 — Proliferative DR | 295 |
| **Total** | **3,662** |

The data was split by image ID into training, validation, and test sets. The reported split was:

| Split | Images |
|---|---:|
| Training | 2,563 |
| Validation | 549 |
| Test | 550 |
| **Total** | **3,662** |

The notebook reported no overlapping image IDs between the three splits.

### Preprocessing notes

The Colab workflow resized retinal images to **224 × 224** and built TensorFlow datasets with image batches shaped `(batch_size, 224, 224, 3)`. Keep local app preprocessing consistent with training preprocessing. Differences in cropping, padding, pixel scaling, or color handling can change predictions.

The dataset archive and individual images are not required to run the already-trained model in the Streamlit app.

## Model training and evaluation

The project used EfficientNetB0 with a classification head, followed by fine-tuning.

Reported training results:

- Initial training best validation accuracy: **71.22%** (as reported by the training log).
- Fine-tuning best validation accuracy: **72.68%**.
- Fine-tuned validation macro F1: **0.5874**.

### Final held-out test results

The final evaluation on the 550-image test set reported:

- **Test accuracy:** 69.09%
- **Test macro F1-score:** 0.5155

Per-class test metrics reported during evaluation:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.953 | 0.974 | 0.964 | 271 |
| Mild | 0.500 | 0.554 | 0.525 | 56 |
| Moderate | 0.754 | 0.327 | 0.456 | 150 |
| Severe | 0.204 | 0.379 | 0.265 | 29 |
| Proliferative DR | 0.272 | 0.568 | 0.368 | 44 |

These results show uneven performance across severity classes. In particular, the reported recall for Severe DR was 37.9%, meaning many severe cases were missed in this evaluation. The model is **not suitable for clinical screening or clinical decisions** without substantial further validation and appropriate clinical oversight.

## Project structure

```text
explainable-dr-screening/
├── app.py
├── pyproject.toml
├── uv.lock                 # Created by `uv sync`, if using uv
├── .env                    # Local secret; do not commit
├── .env.example
├── .gitignore
├── README.md
└── models/
    └── aptos_efficientnet_finetuned.keras
```

Place the fine-tuned model file at `models/aptos_efficientnet_finetuned.keras`. The exact model filename and path in `app.py` must match the file downloaded from Google Drive. The full 9.51 GB dataset is not required to run inference.

## Requirements

- Python **3.11 or 3.12**, using a version supported by your TensorFlow release.
- `uv` package manager (recommended for this project's `pyproject.toml`) or another Python environment manager.
- The trained model file `aptos_efficientnet_finetuned.keras`.
- Internet access only for Groq-generated reports; local inference and Grad-CAM can run offline once dependencies and model are installed.
- A Groq API key for the optional GenAI report.

Typical dependencies:

- `streamlit`
- `tensorflow`
- `numpy`
- `pillow`
- `matplotlib`
- `opencv-python-headless` (if used by the app)
- `openai` (SDK used with Groq's OpenAI-compatible endpoint)
- `python-dotenv`

Ensure versions in `pyproject.toml` are compatible with your installed Python and TensorFlow.

## Setup

### 1. Open the project folder

Open `explainable-dr-screening/` in VS Code.

### 2. Add the trained model

Download `aptos_efficientnet_finetuned.keras` from Google Drive and place it in the `models/` folder. Confirm that the path in `app.py` points to this file.

### 3. Install `uv` if needed

```bash
pip install uv
```

### 4. Install dependencies

From the project root:

```bash
uv sync
```

If you are not using `uv`, create and activate a virtual environment and install the dependencies listed in `pyproject.toml` with your preferred package manager.

### 5. Configure the Groq API key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Replace the placeholder with your own Groq API key. Do not hardcode the real key in source code, commit it to Git, or share it in screenshots. Groq API requests require internet access and may be subject to provider quotas or charges.

Create `.env.example` with placeholders only:

```dotenv
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
```

### 6. Protect secrets and local files

Example `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.streamlit/secrets.toml
```

Model files can be large. Consider separate model storage or Git LFS if sharing the model; do not accidentally commit private keys or large dataset archives.

## Run the Streamlit app

From the project root:

```bash
uv run streamlit run app.py
```

Or, if your environment is already activated and dependencies are installed:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in your browser.

### Troubleshooting

1. Check the VS Code terminal for the traceback if the page is blank or shows an error.
2. Confirm the model file exists at the path expected by `app.py`.
3. Confirm all dependencies installed successfully.
4. If Grad-CAM fails with `matplotlib.cm.get_cmap`, use `matplotlib.colormaps["jet"]` with recent Matplotlib versions.
5. For Groq errors, check the `.env` variable names, internet connection, model availability, and account limits. Local prediction should not depend on a successful GenAI request.

## How the app works

1. **Image upload:** The user uploads a retinal fundus image.
2. **Preprocessing:** The image is converted to RGB and resized/cropped/padded according to the app's preprocessing code.
3. **Prediction:** EfficientNetB0 returns scores for the five DR categories.
4. **Visualization:** The app displays class scores and a Grad-CAM overlay.
5. **GenAI explanation:** If enabled and Groq is available, the app sends the predicted category and relevant score context to the language model for a plain-language report.
6. **Next steps:** The report should encourage appropriate professional eye evaluation without prescribing treatment or claiming a confirmed diagnosis.

The GenAI component receives model outputs and Grad-CAM context only if the application explicitly sends that information. Unless the code sends the image to a vision-capable model, the text model must not claim it examined the retinal image or independently identified lesions.

## Grad-CAM interpretation

Grad-CAM visualizes image regions that influence a selected model prediction.

- Red/yellow regions generally indicate greater relative influence in the displayed heatmap.
- Blue/darker regions generally indicate lower relative influence.
- These colors **do not prove that a region contains a lesion, bleeding, or diabetic damage**.
- Highlighted borders or background can indicate the model may be relying on unintended image features.

Grad-CAM is an explanation aid, not a clinically validated localization tool. Do not describe highlighted regions as confirmed pathology.

## GenAI report

The app can use the OpenAI Python SDK configured for Groq's OpenAI-compatible endpoint. The language model should explain the predicted category in clear English or Hinglish and provide general, cautious next-step guidance.

The prompt should ensure that the model:

- Does not claim to be a doctor or provide a definitive diagnosis.
- Does not invent retinal findings or claim to have seen the image if it receives scores only.
- Does not present uncalibrated model scores as clinical confidence.
- Explains Grad-CAM colors as relative model influence, not confirmed disease locations.
- Advises evaluation by a qualified eye-care professional.
- Avoids medication recommendations and treatment instructions.
- States that the system is an experimental, unvalidated research prototype.

If Groq is unavailable or there is no internet connection, the app should handle the error gracefully. For fully offline reports, implement a local rule-based explanation instead of calling Groq.

## Known limitations

- Test accuracy and macro F1 are modest, and class performance is uneven.
- Severe DR recall was 37.9% in the reported test evaluation.
- Model scores may not be calibrated probabilities.
- This experiment does not establish performance across all cameras, populations, image qualities, or rural clinical settings.
- Grad-CAM does not verify clinical findings and may highlight irrelevant image regions.
- GenAI can produce incorrect or misleading text; output requires careful constraints and must not replace professional judgment.
- The system has not been clinically validated, externally validated, or approved as a medical device.

## Future work

- Test on an independent dataset and real-world acquisition settings, with appropriate permissions.
- Investigate class imbalance and improve recall for severe categories without contaminating the held-out test set.
- Assess calibration and class-wise sensitivity/specificity on a sufficiently large, representative evaluation set.
- Review Grad-CAM behavior for border/background dependence.
- Add automated tests for preprocessing parity between training and inference.
- Add a safe fallback report when Groq is offline.
- Package the app for distribution only after reproducible testing and documentation.

## Acknowledgements

- Dataset: APTOS 2019 Blindness Detection.
- Image classifier: EfficientNetB0.
- Explainability: Grad-CAM.
- Web interface: Streamlit.
- Optional language-model integration: Groq API through the OpenAI Python SDK.

---

**Project status:** Hackathon research prototype. Not for diagnosis, treatment decisions, or replacement of a qualified eye-care professional.
