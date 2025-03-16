# Credit-card-fraud-detection
This project implements a machine learning-based credit card fraud detection system designed to identify fraudulent transactions in real-time. The system uses a trained model to analyze transaction features and classify them as either fraudulent or legitimate.

Key Features
Real-time Fraud Detection: Predicts fraud for incoming transactions.

Machine Learning Model: Uses a trained RandomForestClassifier for predictions.

API Integration: Provides a REST API for seamless integration with other systems.

Interactive Frontend: A React-based web interface for testing and visualization.

Handles Imbalanced Data: Uses SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance.

System Architecture
The system consists of three main components:

Backend (FastAPI):

Handles prediction requests.

Processes transaction data.

Returns fraud predictions.

Frontend (React):

Provides an interactive interface for users.

Displays prediction results.

Machine Learning Model:

Trained on historical transaction data.

Uses features like Time, Amount, and PCA-transformed features (V1 to V28).

Dataset
The system is trained on the Kaggle Credit Card Fraud Detection Dataset, which contains:

284,807 transactions (492 fraudulent).

Features: Time, Amount, and PCA-transformed features (V1 to V28).

Target Variable: Class (0 = Legitimate, 1 = Fraudulent).

How It Works
Data Preprocessing:

Handles missing values.

Scales features using StandardScaler.

Balances the dataset using SMOTE.

Model Training:

Uses a RandomForestClassifier.

Hyperparameters tuned using GridSearchCV.

Prediction:

The trained model predicts whether a transaction is fraudulent.

Results are returned via the API.

Frontend Interaction:

Users input transaction data through the web interface.

The frontend sends the data to the backend API and displays the prediction.

Installation
Prerequisites
Python 3.8+

Node.js (for the frontend)

Required Python packages: fastapi, scikit-learn, pandas, joblib, imbalanced-learn

Steps
Clone the Repository:

bash
Copy
git clone https://github.com/Zoeayilara/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Set Up the Backend:

bash
Copy
cd backend
pip install -r requirements.txt
python api.py
Set Up the Frontend:

bash
Copy
cd frontend
npm install
npm start
Access the System:

Backend: http://localhost:8000

Frontend: http://localhost:3000

Usage
Backend API
Endpoint: /predict

Method: POST

Input:

json
Copy
{
"Time": 0.0,
"V1": -1.359807,
"V2": -0.072781,
"V3": 2.536347,
"V4": 1.378155,
"V5": -0.338321,
"V6": 0.462388,
"V7": 0.239599,
"V8": 0.098698,
"V9": 0.363787,
"V10": 0.090794,
"V11": -0.551600,
"V12": -0.617801,
"V13": -0.991390,
"V14": -0.311169,
"V15": 1.468177,
"V16": -0.470401,
"V17": 0.207971,
"V18": 0.025791,
"V19": 0.403993,
"V20": 0.251412,
"V21": -0.018307,
"V22": 0.277838,
"V23": -0.110474,
"V24": 0.066928,
"V25": 0.128539,
"V26": -0.189115,
"V27": 0.133558,
"V28": -0.021053,
"Amount": 149.62
}
Output:

json
Copy
{
"prediction": 1
}
Frontend
Enter transaction data in the form.

Click Predict to see the result.

Results
Accuracy: ~99.9%

Precision: ~0.90

Recall: ~0.80

F1-Score: ~0.85

Future Improvements
Add support for more machine learning models (e.g., XGBoost, LightGBM).

Implement real-time monitoring and alerts.

Deploy the system to the cloud for scalability.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contributing
Contributions are welcome! Please open an issue or submit a pull request.

Contact
For questions or feedback, please contact:

zoeayilara@gmail.com

GitHub: Zoeayilara
