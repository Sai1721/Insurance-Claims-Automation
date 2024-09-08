# Insurance Claims Automation

## Overview
This project automates the First Notice of Loss (FNOL) process using machine learning and Natural Language Processing (NLP). The goal is to predict claim approvals and streamline the claims processing workflow.

## Files
- `fnol_automation.py`: Python script for data preprocessing, model training, evaluation, and saving.
- `claims_data.csv`: Sample dataset containing claim descriptions and approval statuses.
- `fnol_model.pkl`: Trained model for predicting claim approvals.
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer used for text feature extraction.

## Usage
1. Ensure you have Python installed.
2. Install required packages:
   ```bash
   pip install pandas scikit-learn joblib
