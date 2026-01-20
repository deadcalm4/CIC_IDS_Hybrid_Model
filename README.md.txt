# CIC_IDS_Hybrid_Model
A hybrid deep learning and machine learning–based intrusion detection system using LSTM/GRU for sequential feature learning, stacked autoencoders for feature compression, and SVM for final classification, evaluated on the CIC-IDS-2017 benchmark dataset.

# Hybrid Intrusion Detection System using LSTM, GRU, SAE, and SVM

This repository presents an end-to-end **hybrid intrusion detection system (IDS)** developed using the **CIC-IDS-2017** benchmark dataset.  
The proposed framework combines **deep sequential learning**, **unsupervised feature compression**, and **classical machine learning** to achieve robust and high-accuracy network attack detection.


Due to size and licensing constraints, the CIC-IDS-2017 dataset is not included.


---

## 🚀 Project Overview

Modern networks generate large volumes of sequential traffic data, making intrusion detection a challenging task.  
This project addresses the problem by integrating:

- **LSTM & GRU** – to learn temporal patterns in network traffic  
- **Stacked Autoencoder (SAE)** – to compress and denoise learned features  
- **Support Vector Machine (SVM)** – for robust final classification  

The system is evaluated in a **phase-wise manner**, comparing baseline models, partial hybrids, and the full hybrid architecture.

---

## 🧠 Model Architecture (Hybrid Framework)
Raw Network Traffic
↓
Data Preprocessing & Normalization
↓
Sequence Creation (Sliding Window)
↓
LSTM / GRU (Temporal Feature Learning)
↓
Stacked Autoencoder (Feature Compression)
↓
SVM Classifier
↓
Intrusion Detection (Benign / Attack)