# GenoCGC

**GenoCGC: A Hybrid CNN and Graph-based Transformer Model for Genomic Sequence Classification**

## 🧬 Overview

GenoCGC is a deep learning model designed for the classification of genomic sequences. It combines the strengths of Convolutional Neural Networks (CNNs) for capturing local patterns and Graph-based Transformers (Graphormer) for modeling long-range dependencies in DNA sequences.

This repository provides the implementation of GenoCGC as introduced in the following paper:

> **Anas Abu-Doleh**, "GenoCGC: A Hybrid CNN and Graph-based Transformer Model for Genomic Sequence Classification",  
> *2025 12th International Conference on Information Technology (ICIT)*,  
> Biomedical Systems and Informatics Engineering Department, Hijjawi Faculty for Engineering Technology, Yarmouk University, Irbid, Jordan  
> 📧 Email: anas.abudoleh@yu.edu.jo

## 📊 Dataset

The dataset used in this study is publicly available and can be downloaded from the Hugging Face repository at:  
👉 [https://huggingface.co/datasets/InstaDeepAI/nucleotide](https://huggingface.co/datasets/InstaDeepAI/nucleotide)

## 🛠 Requirements

- Python 3.7+
- PyTorch
- NetworkX
- NumPy
- Scikit-learn

Install dependencies with:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

Run the model with:
```bash
python genocgc.py
```

## 📁 Files Included

- `genocgc.py` – Main script implementing the GenoCGC model
- `requirements.txt` – Python dependencies
- `README.md` – Project documentation

## 📃 License

This project is open source and available under the **MIT License**.
