import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

from google.colab import drive
drive.mount('/content/drive')

folder_path = '/content/drive/MyDrive/Fatigue Data' 
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
all_data = pd.DataFrame()

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path, header=None, names=['Time', 'EMG', 'Fatigue'])
    data['Source_File'] = file
    all_data = pd.concat([all_data, data], ignore_index=True)

print(all_data.head())
print(all_data.isnull().sum())
print(all_data['Fatigue'].value_counts())

X = all_data[['Time', 'EMG']].values
y = all_data['Fatigue'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

def build_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

input_shape = (X_train.shape[1], X_train.shape[2])
rnn_model = build_rnn_model(input_shape)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = rnn_model.fit(X_train, y_train, epochs=50, batch_size=16,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

test_loss, test_acc, test_precision, test_recall = rnn_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')

predictions = rnn_model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

conf_matrix = confusion_matrix(y_test, predicted_labels)
print(f"Confusion Matrix:\n{conf_matrix}")
print(classification_report(y_test, predicted_labels))

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Non-Fatigue', 'Fatigue'], yticklabels=['Non-Fatigue', 'Fatigue'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for RNN Model')
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

def predict_fatigue(model):
    while True:
        try:
            time_input = float(input("Enter Time: "))
            emg_input = float(input("Enter EMG: "))
            new_sample = np.array([[time_input, emg_input]])
            new_sample = scaler.transform(new_sample)
            new_sample = new_sample.reshape((1, 1, new_sample.shape[1]))
            prediction = model.predict(new_sample)
            predicted_class = (prediction > 0.5).astype(int)
            print(f'Predicted class for Time: {time_input}, EMG: {emg_input} (0 for Non-Fatigue, 1 for Fatigue): {predicted_class[0][0]}')
        except ValueError:
            print("Invalid input. Please enter numeric values for Time and EMG.")
        cont = input("Do you want to predict another sample? (yes/no): ")
        if cont.lower() != 'yes':
            break
