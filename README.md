# 🧠 Fine-tuning de Modèles de Question-Answering (SQuAD)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![UVSQ](https://img.shields.io/badge/UVSQ-M2%20Datascale-green)

> **Projet Final - M2 Datascale - Fouille de Données** > Université de Versailles Saint-Quentin-en-Yvelines (UVSQ)

## 📖 Description

Ce projet vise à explorer, entraîner (fine-tuner) et déployer des modèles de **Traitement Automatique du Langage Naturel (NLP)** capables de répondre à des questions basées sur un contexte donné (Question-Answering) .

Nous utilisons le jeu de données standard **SQuAD (Stanford Question Answering Dataset)** pour entraîner 3 architectures de modèles pré-entraînés (Transformers). L'objectif est de comparer leurs performances et de fournir une interface utilisateur web interactive pour tester les modèles en temps réel.

## 🎯 Objectifs

1.  **Les Modèles :** Comparaison de trois architectures distinctes : **T5** (Génératif), **ALBERT** (Optimisé) et **BERT** (Haute Performance) .
2.  **Analyse Comparative :** Évaluer les modèles selon les métriques **F1-Score**, **Exact Match (EM)** et le **temps d'inférence** .
3.  **Interface Utilisateur :** Développer une application Web (Streamlit) permettant aux utilisateurs de poser des questions sur leurs propres textes ou fichiers .
4.  **Déploiement :** Rendre l'application accessible via Hugging Face Spaces .

## 🏗 Architecture du Projet

Le projet est structuré pour séparer la phase de recherche (Notebooks) de la phase de production (App Web) :

```text
Projet_SQuAD/
│
├── 📂 data/                  # Données brutes et pré-traitées
├── 📂 notebooks/             # Contient le notebook principal d'analyse et d'entraînement
│   └── main.ipynb
│
├── 📂 models/                # Sauvegarde des modèles fine-tunés (exclus du git via .gitignore)
├── 📂 app/                   # Code de l'application Web pour le déploiement
│   ├── app.py                # Point d'entrée Streamlit
│   └── utils.py              # Fonctions d'inférence
│
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation du projet
```
## 📦 Dépendances et Rôles Techniques

| Bibliothèque | Rôle  |
| :--- | :--- |
| **`transformers`** (Hugging Face) |  Permet de charger les architectures (T5, DistilBERT, RoBERTa), les tokenizers et les pipelines de QA. |
| **`datasets`** | Utilisé pour télécharger et gérer le dataset SQuAD de manière efficace et standardisée. |
| **`torch` (PyTorch)** | Framework de Deep Learning servant de backend pour les calculs tensoriels et l'optimisation. |
| **`streamlit`** | Framework permettant de créer l'interface utilisateur (Frontend) pour la démo interactive. |
| **`evaluate` / `scikit-learn`** | Calcul des métriques de performance (Exact Match, F1-Score) pour valider les résultats. |
| **`pandas` & `matplotlib`** | Manipulation des données et visualisation des résultats comparatifs. |

## 🚀 Guide d'Installation et d'Exécution

Suivez ces étapes pour reproduire l'environnement de développement et lancer l'application localement.

### 1. Cloner le dépôt
```bash
git clone https://github.com/Ciscom224/Projet_SQuAD_M2.git

```
### 2. Créer un environnement virtuel
**Windows :**
```bash
python -m venv venv
.\venv\Scripts\activate
```
**Mac / Linux :**

```bash
python3 -m venv venv
source venv/bin/activate
```
## 🚀 Guide d'Installation et d'Exécution

* **Installation :**
    ```bash
    pip install -r requirements.txt
    ```

* **Entraînement :**
    Exécuter le notebook situé dans `notebooks/` pour générer les modèles fine-tunés.

* **Lancement de l'App :**
    ```bash
    streamlit run app/app.py
    ```
## 📊 Métriques d'Évaluation

Pour évaluer la performance de nos modèles sur le jeu de données SQuAD, nous utilisons les métriques standards suivantes :

* **Exact Match (EM) :** Mesure le pourcentage de prédictions qui correspondent **exactement** à la réponse attendue (mot pour mot). C'est une métrique très stricte (0 ou 1).
* **F1-Score :** Moyenne harmonique de la précision et du rappel. Cette métrique est plus souple et évalue le chevauchement (overlap) des mots entre la réponse prédite et la vérité terrain.
* **Temps d'inférence (Latence) :** Mesure du temps moyen nécessaire au modèle pour générer une réponse. Cette métrique est cruciale pour évaluer la viabilité du déploiement en temps réel sur l'application Web.

## 👥 Auteurs
* **[Prénom Nom]** – [GitHub](https://github.com/) | [LinkedIn](https://www.linkedin.com/)
* **[Prénom Nom]** – [GitHub](https://github.com/) | [LinkedIn](https://www.linkedin.com/)
* **Mamadou Cissé**  – [GitHub](https://github.com/Ciscom224) | [LinkedIn](https://www.linkedin.com/in/cissemamadou/)


**Encadrant :**
* [Nom du Professeur]