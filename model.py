
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
import logging
import signal
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Signal handler for graceful exit
def signal_handler(sig, frame):
    logging.info("Interrupt received. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Step 1: Load the dataset
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"The file '{filepath}' was not found.")
        exit()

# Step 2: Explore the dataset
def explore_data(df):
    logging.info("Dataset Overview:")
    print(df.head())
    print(df.info())
    print(df.describe())

    logging.info("Checking for missing values:")
    print(df.isnull().sum())

    logging.info("Plotting Class Distribution:")
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution')
    plt.show()

    logging.info("Plotting Correlation Matrix:")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Step 3: Preprocess the data
def preprocess_data(df, target_column):
    logging.info("Preprocessing Data...")
    # Drop rows with NaN in the target column
    df_cleaned = df.dropna(subset=[target_column])
    X = df_cleaned.drop(target_column, axis=1)
    y = df_cleaned[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    logging.info("SMOTE applied successfully.")

    return X_train, X_test, y_train, y_test, df.drop(target_column, axis=1).columns, scaler


# Step 4: Train and evaluate multiple models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, scaler):
    logging.info("Training and Evaluating Models...")

    # Define models to compare
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=1, eval_metric='logloss'),  # Removed use_label_encoder
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=1)
    }

    # Define parameter grids for GridSearchCV
    param_grids = {
        'RandomForest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__max_depth': [4, 6],
            'classifier__criterion': ['gini', 'entropy']
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [4, 6],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8, 1.0]
        },
        'LightGBM': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [4, 6],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8, 1.0]
        }
    }

    results = {}

    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")

        # Define a pipeline without scaling (since data is already scaled)
        pipeline = Pipeline([
            ('classifier', model)
        ])

        # Perform GridSearchCV
        try:
            grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=3, scoring='f1', n_jobs=1)
            grid_search.fit(X_train, y_train)
            logging.info(f"Best Parameters for {model_name}: {grid_search.best_params_}")
            logging.info(f"Best F1 Score for {model_name}: {grid_search.best_score_}")
        except Exception as e:
            logging.error(f"Error during GridSearchCV for {model_name}: {e}")
            continue

        # Evaluate the tuned model
        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test)
        y_pred_proba = best_clf.predict_proba(X_test)[:, 1]

        # Store results
        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importances': best_clf.named_steps['classifier'].feature_importances_ if hasattr(best_clf.named_steps['classifier'], 'feature_importances_') else None
        }

        # Save the model
        joblib.dump(best_clf, f'{model_name}_model.pkl')
        logging.info(f"{model_name} model saved.")

    # Compare results
    logging.info("Model Comparison:")
    for model_name, metrics in results.items():
        logging.info(f"--- {model_name} ---")
        logging.info(f"Accuracy: {metrics['accuracy']}")
        logging.info(f"Precision: {metrics['precision']}")
        logging.info(f"Recall: {metrics['recall']}")
        logging.info(f"F1 Score: {metrics['f1']}")
        logging.info(f"ROC AUC: {metrics['roc_auc']}")
        logging.info("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        logging.info("Classification Report:")
        print(metrics['classification_report'])

    # Save scaler and feature names
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names.tolist(), 'feature_names.pkl')
    logging.info("Scaler and feature names saved.")

# Main function
# Main function
def main():
    # Load the dataset
    df = load_data('creditcard.csv')

    # Use a smaller subset of the data for testing
    df = df.sample(frac=0.1, random_state=42)

    # Explore the dataset
    explore_data(df)

    # Visualizations for Amount by Class
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    f.suptitle('Amount per transaction by class')
    bins = 50
    ax1.hist(fraud['Amount'], bins=bins, color='red')
    ax1.set_title('Fraud')
    ax1.set_ylabel('Number of Transactions')
    ax2.hist(normal['Amount'], bins=bins, color='green')
    ax2.set_title('Normal')
    ax2.set_xlabel('Amount ($)')
    ax2.set_ylabel('Number of Transactions')
    plt.xlim(0, 20000)
    plt.yscale('log')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(fraud['Amount'], bins=50, color='red', alpha=0.5, label='Fraud')
    plt.hist(normal['Amount'], bins=50, color='green', alpha=0.5, label='Normal')
    plt.title('Amount per Transaction by Class')
    plt.xlabel('Amount ($)')
    plt.ylabel('Number of Transactions')
    plt.xlim(0, 20000)
    plt.yscale('log')
    plt.legend()
    plt.show()

    # Preprocess the data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df, 'Class')

    # Train and evaluate multiple models
    train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, scaler)

if __name__ == "__main__":
    main()