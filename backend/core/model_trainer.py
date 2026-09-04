import os
import io
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib



class ModelTrainer:
    def __init__(self, target_column="Churn"):
        self.target_column = target_column
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = []

    def preprocess_data(self, df):
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
            
        df = df.replace(' ', np.nan)
        df = df.dropna()

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].apply(lambda x: 1 if str(x).lower() == "yes" else 0)

        X_encoded = pd.get_dummies(X, drop_first=True)
        self.feature_columns = X_encoded.columns.tolist()
        return X_encoded, y
        
    def train_and_evaluate(self, csv_string):
        df = pd.read_csv(io.StringIO(csv_string))
        X, y = self.preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')

        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        results = {
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0))
        }

        importances = self.model.feature_importances_
        feat_imp = sorted(zip(self.feature_columns, importances), key=lambda x: x[1], reverse=True)
        results["feature_importances"] = {name: float(imp) for name, imp in feat_imp[:10]}

        self.save_model()
        return results

    def save_model(self):
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
        os.makedirs(models_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(models_dir, "rf_model.joblib"))
        joblib.dump(self.scaler, os.path.join(models_dir, "scaler.joblib"))
        joblib.dump(self.feature_columns, os.path.join(models_dir, "features.joblib"))

