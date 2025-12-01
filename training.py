# training.py
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CORRECTED: Keras/TensorFlow Imports for LSTM ---
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def save_report(name, y_test, y_pred):
    report = classification_report(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report_path = f'C:\\MCA\\4th_sem\\Phase2\\project\\model\\{name}_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"{name} report saved.")


def main():
    print("📦 Loading preprocessed data...")
    # Load dataframes with column names
    X_train_df = pd.read_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\X_train.csv', header=None, names=['text'])
    X_test_df = pd.read_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\X_test.csv', header=None, names=['text'])
    y_train_df = pd.read_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\y_train.csv', header=None, names=['label'])
    y_test_df = pd.read_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\y_test.csv', header=None, names=['label'])

    # Combine and clean to ensure alignment and remove empty lines
    train_df = pd.concat([X_train_df, y_train_df], axis=1).dropna()
    test_df = pd.concat([X_test_df, y_test_df], axis=1).dropna()

    # Separate features and labels
    X_train = train_df['text'].astype(str)
    y_train = train_df['label']
    X_test = test_df['text'].astype(str)
    y_test = test_df['label']
    
    print(f"✅ Loaded and aligned {len(X_train)} training samples and {len(X_test)} testing samples.")
    print("🧮 Class distribution:\n", y_train.value_counts())

    # --- Classical Models Pipeline (TF-IDF + SMOTE) ---
    print("\n--- Starting Classical Model Training ---")
    print("🔤 Applying TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    print("📈 Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.1)
    X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train)
    print("✅ Resampled class distribution:\n", pd.Series(y_resampled).value_counts())

    # --- Train Random Forest ---
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_resampled, y_resampled)
    y_pred_rf = rf.predict(X_test_vec)
    save_report("rf", y_test, y_pred_rf)

    # --- Train XGBoost ---
    print("\n⚡ Training XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_resampled, y_resampled)
    y_pred_xgb = xgb.predict(X_test_vec)
    save_report("xgb", y_test, y_pred_xgb)

    # --- Train LightGBM ---
    print("\n💡 Training LightGBM...")
    lgbm = LGBMClassifier(random_state=42)
    lgbm.fit(X_resampled, y_resampled)
    y_pred_lgbm = lgbm.predict(X_test_vec)
    save_report("lgbm", y_test, y_pred_lgbm)

    # --- LSTM Model Pipeline (Load Pre-trained Model) ---
    print("\n--- Starting LSTM Model Evaluation ---")
    MAX_LEN = 300
    
    print(" Loading pre-trained LSTM model and tokenizer...")
    # Assumes these files are in the same directory as the script
    lstm_model = load_model('lstm_model.keras')
    tokenizer = joblib.load('lstm_tokenizer.pkl')
    
    print("Tokenizing and padding sequences for LSTM...")
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)
    
    print(" Making predictions with LSTM model...")
    lstm_probs = lstm_model.predict(X_test_pad).squeeze()
    y_pred_lstm = (lstm_probs > 0.5).astype(int)
    save_report("lstm", y_test, y_pred_lstm)
    
    # --- Save Trained Classical Models ---
    print("\n Saving classical models...")
    model_path = 'C:\\MCA\\4th_sem\\Phase2\\project\\model'
    joblib.dump(rf,   os.path.join(model_path, 'fake_job_model.pkl'))
    joblib.dump(xgb,  os.path.join(model_path, 'fake_job_model_xgb.pkl'))
    joblib.dump(lgbm, os.path.join(model_path, 'fake_job_model_lgbm.pkl'))
    joblib.dump(tfidf, os.path.join(model_path, 'tfidf_vectorizer.pkl'))
    
    # --- Save Final F1 score comparison ---
    print("\n Creating final comparison report...")
    metrics = {
        "RandomForest_F1": f1_score(y_test, y_pred_rf, average='macro'),
        "XGBoost_F1": f1_score(y_test, y_pred_xgb, average='macro'),
        "LightGBM_F1": f1_score(y_test, y_pred_lgbm, average='macro'),
        "LSTM_F1": f1_score(y_test, y_pred_lstm, average='macro')
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(model_path, "comparison_metrics.csv"), index=False)
    print("✅ Model training and evaluation complete. All models and metrics saved.")


if __name__ == "__main__":
    main()
