import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
import pickle
from projectpro import checkpoint
from ML_Pipeline.processing import preprocess_data
from ML_Pipeline.modeling import train_models, evaluate_model, save_models

csv_file_path = "../data/raw/Marketing_Data.csv"
df = pd.read_csv(csv_file_path)
print("Data Read")
checkpoint("fcMar1")

df = preprocess_data(df)
print("Data Processed")

# Model training
X = df[["Lead Owner", "What do you do currently ?", "Marketing Source", "Creation Source", "hour_of_day", "day_of_week"]]
y = df["Interest Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf, xgb, lgb, X_test, y_test = train_models(X, y)

evaluate_model("Random Forest", rf, rf.predict(X_test), X_test, y_test)
evaluate_model("XGBoost", xgb, xgb.predict(X_test), X_test, y_test)
evaluate_model("Light GBM", lgb, lgb.predict(X_test), X_test, y_test)

save_models(rf, xgb, lgb)
checkpoint("fcMar1")