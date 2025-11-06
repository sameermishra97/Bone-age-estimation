# Bone Age Estimation using CNN (Regression)
# Author: Sameer Mishra
# IIT Madras - Data Science & Applications

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Paths
train_dir = '/content/drive/MyDrive/BoneAge/train'
val_dir = '/content/drive/MyDrive/BoneAge/val'

# Data preprocessing
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=(224,224), batch_size=16, class_mode='sparse'
)

val_data = val_gen.flow_from_directory(
    val_dir, target_size=(224,224), batch_size=16, class_mode='sparse'
)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='linear')  # Regression output
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Training
history = model.fit(train_data, validation_data=val_data, epochs=20)

# Plot
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.legend()
plt.title("Bone Age Estimation Performance")
plt.show()

# Save model
model.save('/content/bone_age_model.h5')
