# IMDb Sentiment Classification with DistilBERT

A hands-on learning project to understand how Transformer fine-tuning works from the ground up.

Instead of using a high-level training API, this project implements the core fine-tuning loop manually using PyTorch and Hugging Face Transformers.

## 🚀 Overview

The project fine-tunes `distilbert-base-uncased` on the IMDb movie review dataset to classify reviews as:

- `0` → Negative
- `1` → Positive

The model was trained on a Kaggle NVIDIA Tesla T4 GPU.

## 🧠 What I Learned

This project was built to understand the mechanics behind Transformer fine-tuning:

- Loading and exploring a dataset with Hugging Face `datasets`
- Tokenization with `AutoTokenizer`
- Token IDs and attention masks
- Embedding layers inside Transformer models
- How DistilBERT processes contextual representations
- Classification heads and logits
- Cross-entropy loss
- Backpropagation and gradients
- AdamW optimization
- Batch training with PyTorch `DataLoader`
- Dynamic padding with `DataCollatorWithPadding`
- GPU training with CUDA
- Evaluating a fine-tuned model on unseen data

## 🔄 Training Pipeline

```text
IMDb Reviews
     ↓
AutoTokenizer
     ↓
Token IDs + Attention Masks
     ↓
PyTorch Batches
     ↓
DistilBERT
     ↓
Classification Head
     ↓
Logits
     ↓
Loss
     ↓
Backpropagation
     ↓
AdamW
     ↓
Updated Weights
     ↓
Repeat for Multiple Batches / Epochs

## ⚙️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- CUDA
- Kaggle GPU

## 📊 Training Configuration

| Parameter | Value |
|---|---|
| Model | DistilBERT (`distilbert-base-uncased`) |
| Dataset | IMDb |
| Training samples | 25,000 |
| Test samples | 25,000 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Optimizer | AdamW |
| GPU | NVIDIA Tesla T4 |

## 📈 Results

| Epoch | Average Training Loss |
|---|---|
| 1 | 0.2571 |
| 2 | 0.1352 |
| 3 | 0.0684 |

### Test Accuracy

**92.61%**

## 📁 Project Structure

```text
sentiment-finetuning/
│
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
    └── training-results.png
```

## ▶️ Running Locally

Clone the repository:

```bash
git clone https://github.com/GazalThakur/sentiment-finetuning.git
cd sentiment-finetuning
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py
```

The script automatically uses CUDA when a compatible GPU is available and falls back to CPU otherwise.

## 🎯 Project Goal

This project was primarily built as a learning exercise to understand what actually happens during Transformer fine-tuning — from tokenization and embeddings all the way to gradients, optimizer updates, and evaluation.