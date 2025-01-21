import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

# Step 1: Load the dataset
file_path = 'D:/dump/NNDL/Traffic_accidents/traffic_accidents.csv'

data = pd.read_csv(file_path)

# Step 2: Define the target and features
target_column = 'most_severe_injury'  # Classification target
features = data.drop(columns=[target_column])  # Exclude target from features

# Step 3: Preprocess categorical features using Label Encoding
categorical_cols = features.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le

# Step 4: Preprocess the target column using Label Encoding
target_encoder = LabelEncoder()
data[target_column] = target_encoder.fit_transform(data[target_column])
target = data[target_column]

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Step 6: Scale numerical features
scaler = StandardScaler()
numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Step 7: Build a TensorFlow Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(target_encoder.classes_), activation='softmax')  # Number of classes
])

# Step 8: Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 9: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Step 10: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Step 11: Save the model for future use
model.save('traffic_accident_severity_model.h5')
