import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\fake_job_postings.csv')

# Drop header row if accidentally duplicated
df = df[df['fraudulent'] != 'fraudulent']

# Convert 'fraudulent' column to numeric (handle errors)
df['fraudulent'] = pd.to_numeric(df['fraudulent'], errors='coerce')
df = df.dropna(subset=['fraudulent'])
df['fraudulent'] = df['fraudulent'].astype(int)

# Combine important text fields into one
df['full_text'] = df['title'].fillna('') + ' ' + \
                  df['company_profile'].fillna('') + ' ' + \
                  df['description'].fillna('') + ' ' + \
                  df['requirements'].fillna('') + ' ' + \
                  df['benefits'].fillna('')

# Clean text: lowercase, remove symbols, normalize spaces
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df['full_text'] = df['full_text'].apply(clean_text)

# ✅ Save cleaned full dataset
os.makedirs('C:\\MCA\\4th_sem\\Phase2\\project\\data', exist_ok=True)
df[['full_text', 'fraudulent']].to_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\cleaned_dataset.csv', index=False)

# Train-test split
X = df['full_text']
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save splits for classical models
X_train.to_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\X_train.csv', index=False, header=False)
X_test.to_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\X_test.csv', index=False, header=False)
y_train.to_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\y_train.csv', index=False, header=False)
y_test.to_csv('C:\\MCA\\4th_sem\\Phase2\\project\\data\\y_test.csv', index=False, header=False)

print("✅ Preprocessing complete. Cleaned dataset and train/test files saved.")
