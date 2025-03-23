# fraud-detection-system
Real-time credit card fraud detection using LightGBM and FastAPI | Handles extreme class imbalance | API-ready ML project

This system is a high-quality machine learning pipeline tailored for identifying fraudulent credit card transactions with a strong focus on recall. It incorporates sophisticated data preprocessing, model optimization, and real-time prediction APIs to emulate the construction of an effective fraud detection solution for financial organizations.

---

##  Project Summary
> **"Catch the fraud before it causes damage."**

Credit card fraud presents a significant risk to the financial sector, resulting in billions lost each year. This project illustrates the application of machine learning in a practical, deployable, and scalable manner to detect fraud in real-time. It highlights the entire ML engineering process—from handling and preparing imbalanced data to training, assessing, and deploying a model using FastAPI.

---

## Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Features**: 30 anonymized principal components (V1–V28), Time, and Amount
- **Target**: `Class` (0 = legitimate, 1 = fraudulent)

---

## Key Features

- Uses **Dask** to simulate scalable data ingestion  
- Addresses **577:1 class imbalance** using LightGBM’s `scale_pos_weight`  
- **Custom threshold tuning** to prioritize high recall  
- Evaluates model using **AUC, F1-score, Precision, Recall**  
- Deploys model using **FastAPI** with JSON POST endpoint  
- Includes **real-time fraud probability response**

---

## Tech Stack

- **Language**: Python 3.11  
- **Modeling**: LightGBM, scikit-learn  
- **Data Handling**: Dask, Pandas, NumPy  
- **Deployment**: FastAPI, Uvicorn  
- **Testing**: Requests, Swagger UI  
- **Dev Tools**: Colab, GitHub, Ngrok

---

## Model Results

| Metric         | Value   |
|----------------|---------|
| ROC-AUC Score  | **0.905** |
| Recall (Fraud) | **87%** |
| Precision      | 9%     |
| F1 Score       | 0.16   |

> High recall was prioritized to reduce false negatives in fraud detection.

---

## API Usage

### POST `/predict`

Send a JSON payload with transaction features. Example input:

```json
{
  "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155, "V5": -0.338321,
  "V6": 0.462388, "V7": 0.239599, "V8": 0.098698, "V9": 0.363787, "V10": 0.090794,
  "V11": -0.5516, "V12": -0.617801, "V13": -0.99139, "V14": -0.311169, "V15": 1.468177,
  "V16": -0.4704, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
  "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928, "V25": 0.128539,
  "V26": -0.189115, "V27": 0.133558, "V28": -0.021053, "Amount": 149.62
}
```

Example response:

```json
{
  "fraud_probability": 0.8731
}
```

---

## How to Run This Project

1. **Clone the repository**
```bash
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python src/train.py
```

4. **Start the FastAPI server**
```bash
uvicorn src.deploy.app:app --reload
```

---

## Project Structure

```
fraud-detection-system/
├── data/                    # Optional sample data or Kaggle link
├── notebooks/               # EDA and modeling notebooks
├── src/
│   ├── train.py             # Model training and evaluation
│   └── deploy/
│       └── app.py           # FastAPI service
├── lgb_model.pkl            # Trained LightGBM model
├── scaler.pkl               # Trained scaler
├── requirements.txt         # All dependencies
└── README.md                # You are here
```

---

## Lessons Learned

- How to manage **extremely imbalanced data** without overfitting
- How to tune and evaluate a **classification model for rare events**
- How to design a **real-time prediction API**
- How to structure a **deployable, professional-grade ML project**

---

## Contact

**G. Kablan Assebian**
_MSc Data Science Candidate | Entry-Level Data Scientist | Future ML Engineer_

LinkedIn • gomis.k.assebian@gmail.com

---

© 2025 G. Kablan Assebian. MIT License.


