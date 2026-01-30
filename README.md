# 🧠 Fine-tuning de Modèles de Question-Answering (SQuAD)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![UVSQ](https://img.shields.io/badge/UVSQ-M2%20Datascale-green)

> **Projet Final - M2 Datascale - Fouille de Données** > Université de Versailles Saint-Quentin-en-Yvelines (UVSQ)

## 📖 Description

Ce projet vise à explorer, entraîner (fine-tuner) et déployer des modèles de **Traitement Automatique du Langage Naturel (NLP)** capables de répondre à des questions basées sur un contexte donné (Question-Answering).

Nous utilisons le jeu de données standard **SQuAD (Stanford Question Answering Dataset)** [cite: 2] pour entraîner plusieurs architectures de modèles pré-entraînés (Transformers). L'objectif est de comparer leurs performances et de fournir une interface utilisateur web interactive pour tester les modèles en temps réel.

## 🎯 Objectifs

1.  **Fine-tuning :** Entraîner au moins 3 modèles (ex: DistilBERT, RoBERTa, CamemBERT) sur SQuAD[cite: 4].
2.  **Analyse Comparative :** Évaluer les modèles selon les métriques F1-Score, Exact Match (EM) et le temps d'inférence[cite: 6].
3.  **Interface Utilisateur :** Développer une application Web (Streamlit/FastAPI) permettant aux utilisateurs de poser des questions sur leurs propres textes ou fichiers.
4.  **Déploiement :** Rendre l'application accessible via Hugging Face Spaces[cite: 7, 14].

## 🏗 Architecture du Projet

Le projet est structuré pour séparer la phase de recherche (Notebooks) de la phase de production (App Web) :

```text
Projet_SQuAD/
│
├── 📂 data/                  # Données brutes et pré-traitées
├── 📂 notebooks/             # Contient le notebook principal d'analyse et d'entraînement
│   └── Master_Finetuning_Analysis.ipynb
│
├── 📂 models/                # Sauvegarde des modèles fine-tunés (exclus du git via .gitignore)
├── 📂 app/                   # Code de l'application Web pour le déploiement
│   ├── app.py                # Point d'entrée Streamlit
│   └── utils.py              # Fonctions d'inférence
│
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation du projet