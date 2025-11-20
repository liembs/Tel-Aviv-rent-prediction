import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# ----- ElasticNet Wrapper -----
class ElasticNetWrapper:
    def __init__(self, df, target_column='price', X_test=None, y_test=None):
        features_to_drop = ['garden_area', 'has_parking', 'has_storage', 'room_num', 'ac']
        df = df.drop(columns=features_to_drop, errors='ignore')
        y_train = df[target_column]
        X_train = df.drop(columns=[target_column])
        self.feature_names = X_train.columns.tolist()
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        print("📌 ElasticNet Coefficients:", self.model.coef_)
        print("📌 ElasticNet Intercept:", self.model.intercept_)

        if X_test is not None and y_test is not None:
            X_test = X_test.drop(columns=features_to_drop, errors='ignore')
            for col in self.feature_names:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[self.feature_names]
            X_test_scaled = self.scaler.transform(X_test)
            preds = self.model.predict(X_test_scaled)
            r2 = self.model.score(X_test_scaled, y_test)
            rmse = np.sqrt(np.mean((y_test - preds) ** 2))
            print(f"📈 ElasticNet R² על סט הבדיקה: {r2:.2f}")
            print(f"📊 ElasticNet RMSE על סט הבדיקה: {rmse:.2f}")

    def predict(self, X_new):
        features_to_drop = ['garden_area', 'has_parking', 'has_storage', 'room_num', 'ac']
        X_new = X_new.drop(columns=features_to_drop, errors='ignore')
        for col in self.feature_names:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[self.feature_names]
        X_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_scaled)

# ----- DecisionTree Wrapper -----
class DecisionTreeWrapper:
    def __init__(self, df, target_column='price', X_test=None, y_test=None):
        features_to_drop = ['garden_area', 'has_parking', 'has_storage', 'room_num', 'ac']
        df = df.drop(columns=features_to_drop, errors='ignore')
        y_train = df[target_column]
        X_train = df.drop(columns=[target_column])
        self.feature_names = X_train.columns.tolist()
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = DecisionTreeRegressor(random_state=42, max_depth=5)
        self.model.fit(X_train_scaled, y_train)
        print("📌 DecisionTree Feature Importances:", self.model.feature_importances_)

        if X_test is not None and y_test is not None:
            X_test = X_test.drop(columns=features_to_drop, errors='ignore')
            for col in self.feature_names:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[self.feature_names]
            X_test_scaled = self.scaler.transform(X_test)
            preds = self.model.predict(X_test_scaled)
            r2 = self.model.score(X_test_scaled, y_test)
            rmse = np.sqrt(np.mean((y_test - preds) ** 2))
            print(f"📈 DecisionTree R² על סט הבדיקה: {r2:.2f}")
            print(f"📊 DecisionTree RMSE על סט הבדיקה: {rmse:.2f}")

    def predict(self, X_new):
        features_to_drop = ['garden_area', 'has_parking', 'has_storage', 'room_num', 'ac']
        X_new = X_new.drop(columns=features_to_drop, errors='ignore')
        for col in self.feature_names:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[self.feature_names]
        X_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_scaled)

# ----- Combined Model Wrapper -----
class CombinedModelWrapper:
    def __init__(self, df, target_column='price', X_test=None, y_test=None):
        self.elastic_model = ElasticNetWrapper(df, target_column, X_test, y_test)
        self.tree_model = DecisionTreeWrapper(df, target_column, X_test, y_test)

    def predict(self, X_new):
        preds_elastic = self.elastic_model.predict(X_new)
        preds_tree = self.tree_model.predict(X_new)
        return {
            'ElasticNet': preds_elastic,
            'DecisionTree': preds_tree
        }

# ===== אימון ושמירת המודל (רק כשמריצים את הקובץ ישירות) =====
if __name__ == "__main__":
    from assets_data_prep import prepare_data
    df = pd.read_csv("train.csv") 
    df_prepared = prepare_data(df, "train")
    en_model = CombinedModelWrapper(df_prepared)

    import joblib
    joblib.dump(en_model.elastic_model, 'trained_model.pkl')
    print("✅ המודל נשמר כ-trained_model.pkl")
