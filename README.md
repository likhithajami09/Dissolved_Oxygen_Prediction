<div align="center">

# 🌊 Smart & Sustainable Aquaculture
### Predicting Dissolved Oxygen with a Hybrid Deep Learning Ensemble

*Fish can't tell you when the water's running out of oxygen. This model can — before it becomes a problem.*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Accuracy-97.85%25-brightgreen?style=for-the-badge)

</div>

---

## 🐟 The Problem

Dissolved oxygen (DO) is the single most critical factor keeping aquatic life alive — and one of the hardest to track in real time. Traditional monitoring is slow, reactive, and often catches problems *after* the damage is done: stressed fish, disease outbreaks, crashed yields.

**This project flips that script** — predicting DO levels before they become a crisis, using a hybrid deep learning pipeline built for real aquaculture environments.

## ⚡ What Makes This Different

Most existing approaches (Linear Regression, SVM, single LSTM/GRU models) either miss long-term temporal patterns or drown in irrelevant features. This system fixes both problems at once:

| Stage | What it does |
|---|---|
| 🎯 **LightGBM Feature Selection** | Cuts the noise — keeps only the water-quality parameters that actually matter |
| 🧠 **Hybrid Ensemble (RNN + GRU + BiLSTM + Attention)** | Captures both short-term spikes and long-term trends in the data |
| 📊 **Flask + SQLite Web App** | Secure login, data upload, and live testing — not just a notebook experiment |

## 🏆 Results That Speak for Themselves

| Model | RMSE ↓ | MAE ↓ | Accuracy ↑ |
|---|---|---|---|
| LightGBM–LSTM | 0.1531 | 0.1214 | 93.42% |
| LightGBM–GRU | 0.1445 | 0.1105 | 94.11% |
| LightGBM–BiSRU–Attention | 0.1254 | 0.1013 | 96.28% |
| **🥇 Proposed Ensemble Model** | **0.1095** | **0.0907** | **97.85%** |

The proposed hybrid model beats every baseline — lower error, higher accuracy, across the board.

## 🧩 System at a Glance

```
Raw Water Quality Data → Preprocessing → LightGBM Feature Selection
        → Hybrid Deep Learning Ensemble (RNN + GRU + BiLSTM + Attention)
        → Prediction & Evaluation → Flask Dashboard
```

**Modules:** Data Acquisition & Preprocessing · Feature Selection · Prediction & Evaluation (MSE/RMSE/MAE) · Secure Web UI

## 🛠️ Tech Stack

`Python` · `Flask` · `LightGBM` · `RNN` `GRU` `BiLSTM` `Attention` · `Pandas` · `Scikit-learn` · `SQLite` · `HTML/CSS/JS`

## 🚀 Get It Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
python app.py

# 3. Open in your browser
http://127.0.0.1:5000/
```

## 📁 Project Structure

```
Dataset/       → raw & processed water quality data
model/         → trained hybrid ensemble model
templates/     → Flask HTML templates
static/        → CSS/JS assets
uploads/       → user-uploaded data for testing
app.py         → Flask application entry point
requirements.txt
```

## 🌱 Why It Matters

Aquaculture is one of the fastest-growing food industries in the world — and water quality is its biggest bottleneck. A model that predicts oxygen crashes before they happen isn't just a machine learning exercise; it's a tool for **sustainable, higher-yield fish farming**.

---

<div align="center">

⭐ **If this project made you think differently about aquaculture + AI, consider starring the repo!**

</div>
