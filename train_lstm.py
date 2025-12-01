# train_lstm.py
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

# --- Keras/TensorFlow Imports ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    print("📦 Loading preprocessed data...")
    # --- CORRECTED: More robust data loading to prevent sample mismatch ---
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
    # --- END OF CORRECTION ---

    # --- LSTM Data Preparation ---
    MAX_WORDS = 5000  # Maximum number of words to keep in the vocabulary
    MAX_LEN = 300     # Maximum length of a sequence (job description)
    
    print(" Tokenizing and padding sequences for LSTM...")
    # Create and fit the tokenizer on the training text
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences of integers
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad the sequences to ensure they all have the same length
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    # --- Handle Class Imbalance ---
    print("⚖️ Calculating class weights for the model...")
    # This tells the model to pay more attention to the minority class (fake jobs)
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Calculated Class Weights: {class_weights_dict}")
    
    # --- Build and Compile LSTM Model ---
    print("🧠 Building LSTM model architecture...")
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
    model.add(SpatialDropout1D(0.2)) # Helps prevent overfitting
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid')) # Output layer for binary classification
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print(model.summary())
    
    # --- Train the Model ---
    print("\n🧠 Training LSTM model...")
    history = model.fit(
        X_train_pad, 
        y_train, 
        epochs=3,          # Number of passes through the data
        batch_size=64,     # Number of samples per gradient update
        validation_split=0.1, # Use 10% of training data for validation
        class_weight=class_weights_dict # Apply the calculated weights
    )
    
    # --- Evaluate the Model ---
    print("\n📊 Evaluating model on the test set...")
    lstm_probs = model.predict(X_test_pad).squeeze()
    y_pred_lstm = (lstm_probs > 0.5).astype(int)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred_lstm))

    # --- Save the Model and Tokenizer ---
    print("\n💾 Saving the trained model and tokenizer...")
    model.save('C:\\MCA\\4th_sem\\Phase2\\project\\lstm_model.keras')
    joblib.dump(tokenizer, 'C:\\MCA\\4th_sem\\Phase2\\project\\lstm_tokenizer.pkl')
    
    print("✅ LSTM training complete. Model and tokenizer saved.")


if __name__ == "__main__":
    main()

