import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(42)
tf.random.set_seed(42)
MAX_WORDS= 10000
MAX_LENGTH=200
def read_data():
    df = pd.read_csv('ai_human_content_detection_dataset.csv')
    return df


def data_exploration():
    print("=" * 80)
    print("DATASET EXPLORATION")
    print("=" * 80)
    df = read_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

    print("\nFirst 3 samples:")
    print(df[['text_content', 'label']].head(3))

    print("\nLabel distribution:")
    print(df['label'].value_counts())

    # Visualize label distribution
    plt.figure(figsize=(8, 5))
    df['label'].value_counts().plot(kind='bar')
    plt.title('Distribution of AI vs Human Text')
    plt.xlabel('Label (0=Human, 1=AI)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()
    return df


def data_preprocessing():
    df = data_exploration()
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)

    # Extract features and labels
    texts = df['text_content'].values
    labels = df['label'].values
    if labels.dtype == 'object':
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        print(f"\nLabel mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    else:
        print(f"\nLabels are already numeric: {np.unique(labels)}")

    # Split the data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nTraining samples: {len(X_train_text)}")
    print(f"Testing samples: {len(X_test_text)}")
    return X_train_text,X_test_text,y_train,y_test


def text_vectorization():

    X_train_text,X_test_text,y_train,y_test = data_preprocessing()

    print("=" * 80)
    print("TEXT VECTORIZATION")
    print("=" * 80)

    MAX_WORDS = 10000
    MAX_LENGTH = 200
    print(f"\nParameters:")
    print(f"- Max vocabulary size: {MAX_WORDS}")
    print(f"- Max sequence length: {MAX_LENGTH}")

    # Create and fit tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_text)

    # Convert texts to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    # Pad sequences
    X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

    print(f"\nVocabulary size: {len(tokenizer.word_index)}")
    print(f"Padded sequence shape: {X_train_padded.shape}")
    return X_train_padded, X_test_padded, y_train, y_test


def create_basic_model():
    """
        Create a basic neural network model.
        """
    model = models.Sequential([
        # Embedding layer: converts word indices to dense vectors
        layers.Embedding(input_dim=MAX_WORDS,
                         output_dim=128,
                         input_length=MAX_LENGTH),

        # Flatten the 2D embedding to 1D
        layers.Flatten(),

        # Hidden layer with ReLU activation
        layers.Dense(64, activation='relu'),

        # Output layer with sigmoid for binary classification
        layers.Dense(1, activation='sigmoid')

    ])

    model.build(input_shape=(None, MAX_LENGTH))
    return model


# Create and compile the model

def train_basic_model():
    # create n compile the model
    basic_model = create_basic_model()

    basic_model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Basic Model Architecture:")
    basic_model.summary()
    X_train_padded, X_test_padded, y_train, y_test = text_vectorization()

    # Train the basic model
    print("Training Basic Model...")
    print("=" * 50)
    history_basic =basic_model.fit(X_train_padded,y_train, batch_size = 32, epochs=10, validation_split= 0.2, verbose = 1)

    # Evaluate the model
    print("\nEvaluating Basic Model...")
    basic_predictions = (basic_model.predict(X_test_padded) > 0.5).astype(int).flatten()
    basic_accuracy = accuracy_score(y_test, basic_predictions)

    print(f"\n{'=' * 50}")
    print(f"Basic Model Test Accuracy: {basic_accuracy:.4f}")
    print(f"{'=' * 50}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_basic.history['accuracy'], label='Training')
    plt.plot(history_basic.history['val_accuracy'], label='Validation')
    plt.title('Basic Model - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_basic.history['loss'], label='Training')
    plt.plot(history_basic.history['val_loss'], label='Validation')
    plt.title('Basic Model - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def create_dropout_model():

    model = models.Sequential(
        [
            layers.Embedding(input_dim=MAX_WORDS,output_dim=128,input_length=MAX_LENGTH),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128,activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(64,activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1,activation='sigmoid')
        ]
    )
    model.build(input_shape={None,MAX_LENGTH})
    return model

def create_compile_dropout():
    # Create and compile the dropout model
    dropout_model = create_dropout_model()

    dropout_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Dropout Model Architecture:")
    dropout_model.summary()
    return dropout_model

def early_stopping_call_back():
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience = 3,
        restore_best_weights = True
    )

    print("Training Dropout Model...")
    print("=" * 50)
    dropout_model = create_compile_dropout()
    X_train_padded, X_test_padded, y_train, y_test = text_vectorization()
    dropout_model.fit(
        X_train_padded,
        y_train,
        batch_size=32,
        epochs=15,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    # Evaluate dropout model
    print("\nEvaluating Dropout Model...")
    dropout_predictions = (dropout_model.predict(X_test_padded) > 0.5).astype(int).flatten()
    dropout_accuracy = accuracy_score(y_test, dropout_predictions)

    print(f"\n{'=' * 50}")
    print(f"Dropout Model Test Accuracy: {dropout_accuracy:.4f}")
    print(f"{'=' * 50}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, dropout_predictions,
                                target_names=['Human', 'AI']))


def create_batchnorm_model():
    model = models.Sequential(
        [
            layers.Embedding(input_dim=MAX_WORDS,output_dim=128,input_length=MAX_LENGTH),
            layers.GlobalAveragePooling1D(),

            # First dense block with batch normalization
            layers.Dense(128),
            layers.BatchNormalization(),  # Normalize before activation
            layers.Activation('relu'),
            layers.Dropout(0.3),

            # Second dense block with batch normalization
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),

            layers.Dense(1, activation='sigmoid')

        ]
    )

    # Build the model
    model.build(input_shape=(None, MAX_LENGTH))
    return model


def batch_normalisation():
    # Create and compile the batch norm model
    batchnorm_model = create_batchnorm_model()

    batchnorm_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Batch Normalization Model Architecture:")
    batchnorm_model.summary()

    print("Training Batch Normalization Model...")
    print("=" * 50)
    print("Note: This model uses LSTM, so training might be slower")
    X_train_padded, X_test_padded, y_train, y_test = text_vectorization()
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    history_batchnorm = batchnorm_model.fit(
        X_train_padded, y_train,
        batch_size=32,
        epochs=15,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate batch norm model
    print("\nEvaluating Batch Normalization Model...")
    batchnorm_predictions = (batchnorm_model.predict(X_test_padded) > 0.5).astype(int).flatten()
    batchnorm_accuracy = accuracy_score(y_test, batchnorm_predictions)

    print(f"\n{'=' * 50}")
    print(f"Batch Norm Model Test Accuracy: {batchnorm_accuracy:.4f}")
    print(f"{'=' * 50}")
    return batchnorm_model

def predict_text(text, model, tokenizer, max_length=MAX_LENGTH):
    """
    Predict whether a text is AI-generated or human-written.
    """
    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded, verbose=0)[0][0]

    return prediction



if __name__ == '__main__':
    # train_basic_model()
    # create_compile_dropout()
    # early_stopping_call_back()
    batchnorm_model= batch_normalisation()
    tokenizer = Tokenizer()
    print("=" * 80)
    print("TEST WITH CUSTOM TEXTS")
    print("=" * 80)

    # Test texts
    test_texts = [
        "Hello world",
        "In accordance with the aforementioned parameters, we shall proceed to implement the optimal solution.",
        "Hey! Just wanted to say hi and see how you're doing. Hope everything's good!",
        "The implementation demonstrates superior performance metrics across all evaluated dimensions.",
        "LOL that's so funny! Can't believe that happened 😂"
    ]

    print("\nUsing the Advanced Model for predictions:\n")

    for i, text in enumerate(test_texts, 1):
        pred = predict_text(text, batchnorm_model, tokenizer)
        label = "AI-generated" if pred > 0.5 else "Human-written"
        confidence = pred if pred > 0.5 else (1 - pred)

        print(f"Text {i}: '{text[:60]}...'" if len(text) > 60 else f"Text {i}: '{text}'")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 40)






