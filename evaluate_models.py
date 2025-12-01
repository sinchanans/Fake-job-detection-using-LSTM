import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Data ---
X_test_df = pd.read_csv("data/X_test.csv", header=None, names=["text"])
y_test_df = pd.read_csv("data/y_test.csv", header=None, names=["label"])

# Combine and clean to ensure alignment and remove empty lines
test_df = pd.concat([X_test_df, y_test_df], axis=1).dropna()

# Separate features and labels
X_test = test_df['text'].astype(str)
y_test = test_df['label']

# --- Load Models ---
rf = joblib.load("model/fake_job_model.pkl")
xgb = joblib.load("model/fake_job_model_xgb.pkl")
lgbm = joblib.load("model/fake_job_model_lgbm.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

X_test_tfidf = tfidf.transform(X_test)

# --- Load LSTM ---
lstm_model = load_model("lstm_model.keras")
tokenizer = joblib.load("lstm_tokenizer.pkl")
MAX_LEN = 300
X_seq = tokenizer.texts_to_sequences(X_test)
X_padded = pad_sequences(X_seq, maxlen=MAX_LEN)

# --- Evaluation Function ---
metrics = []

def evaluate_model(y_true, y_pred, model_name):
    pos = 1 # for binary labels like 0/1

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos, zero_division=0)

    metrics.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

    print(f"\n📊 {model_name}")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")

# --- Evaluate Each Model ---
evaluate_model(y_test, rf.predict(X_test_tfidf), "Random Forest")
evaluate_model(y_test, xgb.predict(X_test_tfidf), "XGBoost")
evaluate_model(y_test, lgbm.predict(X_test_tfidf), "LightGBM")

lstm_probs = lstm_model.predict(X_padded, verbose=0).squeeze()
lstm_preds = (lstm_probs > 0.5).astype(int)
evaluate_model(y_test, lstm_preds, "LSTM")

# --- Convert to DataFrame ---
df_results = pd.DataFrame(metrics)

# --- Save to CSV ---
df_results.to_csv("model_evaluation_metrics.csv", index=False)
print("\n📄 Saved metrics to 'model_evaluation_metrics.csv'")

# --- Plot Bar Chart (All Scores) ---
plt.figure(figsize=(10, 6))
bar_width = 0.2
x = range(len(df_results))

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
for i, metric in enumerate(metrics_to_plot):
    plt.bar([p + i * bar_width for p in x], df_results[metric], width=bar_width, label=metric)

plt.xticks([p + 1.5 * bar_width for p in x], df_results["Model"])
plt.ylabel("Score")
plt.title("Model Evaluation Metrics")
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison_bar_chart.png")
plt.show()