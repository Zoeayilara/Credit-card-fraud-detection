import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
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

# Step 4: Train and evaluate the model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names, scaler):
    logging.info("Training and Evaluating Model...")

    # Define a pipeline without scaling (since data is already scaled)
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
    ])

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__max_depth': [4, 6],
        'classifier__criterion': ['gini', 'entropy']
    }

    # Perform GridSearchCV
    try:
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=1)
        grid_search.fit(X_train, y_train)
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Best F1 Score: {grid_search.best_score_}")
    except Exception as e:
        logging.error(f"Error during GridSearchCV: {e}")
        exit()

    # Evaluate the tuned model
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    y_pred_proba = best_clf.predict_proba(X_test)[:, 1]

    logging.info("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importances = pd.Series(best_clf.named_steps['classifier'].feature_importances_, index=feature_names)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.show()

    # Save the model and scaler and feature names
    joblib.dump(best_clf, 'credit_card_fraud_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names.tolist(), 'feature_names.pkl')
    logging.info("Feature names saved.")
    logging.info("Model and scaler saved.")

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

    plt.figure(figsize=(12, 8))
    plt.scatter(fraud['id'], fraud['Amount'], color='red', alpha=0.5, label='Fraud')
    plt.scatter(normal['id'], normal['Amount'], color='green', alpha=0.1, label='Normal')
    plt.title('Id Of transaction vs Amount by Class')
    plt.xlabel('Id')
    plt.ylabel('Amount ($)')
    plt.legend()
    plt.show()

    # Preprocess the data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df, 'Class')

    # Train and evaluate the model
    train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names, scaler)

if __name__ == "__main__":
    main()