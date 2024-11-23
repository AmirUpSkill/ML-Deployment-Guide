# app_v1.py
import pandas as pd
from data_loader import load_data
from feature_engineering import scale_features
from model_training import split_data, train_model, evaluate_model, create_pipeline , save_model
from utils import plot_confusion_matrix

def main():
    """Main function to run the ML workflow."""

    # Load Data
    data = load_data()
    print("Data loaded successfully.\n")
    print(data.head())
    print(data.info())
    print(data.describe())

    # Feature Engineering
    X = data.drop('target', axis=1)
    y = data['target']
    X_scaled, scaler = scale_features(X) # Scale the features and return the scaler
    print("\nFeatures scaled successfully.")

    # Split Data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    print("\nData split into training and testing sets.")

    # Train and Evaluate Logistic Regression
    logreg_model = train_model(X_train, y_train, model_type='logistic_regression')
    logreg_pred = logreg_model.predict(X_test)
    save_model(logreg_model,'logreg_model.joblib')
    print("\nLogistic Regression Model:")
    report_lr , matrix_lr = evaluate_model(y_test, logreg_pred)
    plot_confusion_matrix(y_test, logreg_pred, 'Logistic Regression Confusion Matrix')

    # Train and Evaluate k-NN
    knn_model = train_model(X_train, y_train, model_type='knn')
    knn_pred = knn_model.predict(X_test)
    save_model(knn_model,'knn_model.joblib')
    print("\nk-Nearest Neighbors Model:")
    report_knn , matrix_knn = evaluate_model(y_test, knn_pred)
    plot_confusion_matrix(y_test, knn_pred, 'k-NN Confusion Matrix')

    # Train and Evaluate Decision Tree
    dt_model = train_model(X_train, y_train, model_type='decision_tree')
    dt_pred = dt_model.predict(X_test)
    save_model(dt_model,'dt_model.joblib')
    print("\nDecision Tree Model:")
    report_dt , matrix_dt = evaluate_model(y_test, dt_pred)