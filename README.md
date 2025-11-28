# Fraud_Detection_System
Fraud Detection System is a hands-on ML project that turns raw transaction logs into insights, using models like Logistic Regression, Decision Trees, XGBoost, and KNN to spot suspicious activity and help prevent real-world financial fraud.
## Features

- Data preprocessing: handle missing values, one-hot encode transaction types, and scale numerical features (step, amount, balances).  
- Models: Logistic Regression, Decision Tree in the script; XGBoost and KNN additionally explored in the notebook.  
- Evaluation: accuracy, confusion matrices, classification reports, and feature importance plots.  
- Visualizations: transaction type distributions and donut charts to understand data patterns.

## Tech Stack

- Python, Jupyter Notebook  
- pandas, numpy, seaborn, matplotlib  
- scikit-learn, xgboost  

## Getting Started
- git clone https://github.com/ankushgr/Fraud-Detection-System.git
- cd Fraud-Detection-System
- pip install -r requirements.txt
- Place `Fraud_Log.csv` in the project root.
- Run the script:
- python Fraud_Detection_System.py
- Or open the notebook:
- jupyter notebook Fraud_Detection_System.ipynb

## Project Structure

- `Fraud_Detection_System.py` – end-to-end pipeline (preprocessing, training, evaluation).  
- `Fraud_Detection_System.ipynb` – exploratory notebook with extra models and visuals.  
- `requirements.txt` – dependencies.  
- `Fraud_Log.csv` – transaction dataset (not included; add locally).

## Future Improvements

- Better handling of class imbalance.  
- Additional models and hyperparameter tuning.  
- Simple API or web interface for real-time scoring.






