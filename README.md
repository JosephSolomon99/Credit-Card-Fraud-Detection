# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using anomaly detection and classification algorithms.

## Overview

This project implements multiple machine learning approaches to identify fraudulent credit card transactions. Credit card fraud detection is a critical problem in the financial industry, and machine learning models can help identify suspicious patterns in transaction data.

## Features

- Data preprocessing and feature engineering for transaction data
- Implementation of multiple ML algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Neural Networks
- Handling imbalanced datasets using SMOTE and undersampling techniques
- Model evaluation with precision, recall, F1-score, and ROC-AUC
- Visualization of results and feature importance

## Dataset

This project is designed to work with credit card transaction datasets, typically containing:
- Transaction amount
- Time
- Anonymized features (V1, V2, ... V28)
- Class label (0 = legitimate, 1 = fraud)

Common datasets:
- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- European cardholder transactions (September 2013)

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JosephSolomon99/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.fraud_detector import FraudDetector
from src.data_processor import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/creditcard.csv')

# Train model
detector = FraudDetector(model_type='random_forest')
detector.train(X_train, y_train)

# Make predictions
predictions = detector.predict(X_test)

# Evaluate
detector.evaluate(X_test, y_test)
```

## Project Structure

```
Credit-Card-Fraud-Detection/
├── data/                   # Dataset directory (not included in repo)
├── src/                    # Source code
│   ├── fraud_detector.py   # Main model implementation
│   ├── data_processor.py   # Data preprocessing utilities
│   └── utils.py            # Helper functions
├── notebooks/              # Jupyter notebooks for analysis
│   └── exploratory_analysis.ipynb
├── models/                 # Saved trained models
├── results/                # Output results and visualizations
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Methodology

### 1. Data Preprocessing
- Handle missing values
- Feature scaling (StandardScaler)
- Train-test split with stratification

### 2. Handling Class Imbalance
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random undersampling
- Class weight adjustment

### 3. Model Training
- Cross-validation for hyperparameter tuning
- Grid search for optimal parameters
- Ensemble methods for improved performance

### 4. Evaluation Metrics
- Precision: Accuracy of fraud predictions
- Recall: Percentage of actual frauds detected
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Overall model performance
- Confusion Matrix: Detailed classification breakdown

## Results

Model performance on test set:

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| Gradient Boosting | TBD | TBD | TBD | TBD |
| Neural Network | TBD | TBD | TBD | TBD |

*Note: Results will be updated after model training*

## Key Challenges

1. **Highly Imbalanced Data**: Fraud transactions typically represent <1% of all transactions
2. **False Positives**: Minimizing incorrect fraud flags to avoid customer inconvenience
3. **Feature Engineering**: Creating meaningful features from transaction data
4. **Real-time Detection**: Ensuring model can process transactions quickly

## Future Improvements

- [ ] Implement deep learning models (LSTM, Autoencoders)
- [ ] Add real-time prediction API
- [ ] Incorporate temporal patterns and user behavior analysis
- [ ] Deploy model using Docker and cloud services
- [ ] Add model monitoring and retraining pipeline

## Technologies Used

- **Python 3.8+**
- **Scikit-learn**: ML algorithms and preprocessing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Imbalanced-learn**: Handling imbalanced datasets
- **Jupyter**: Interactive analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Joseph Solomon**
- GitHub: [@JosephSolomon99](https://github.com/JosephSolomon99)
- Portfolio: [josephsolomon99.github.io](https://josephsolomon99.github.io)

## Acknowledgments

- Dataset source: Kaggle Credit Card Fraud Detection
- Inspired by real-world financial fraud detection systems
- Built as part of a machine learning portfolio

---

*This project demonstrates practical application of machine learning for fraud detection in financial transactions.*
