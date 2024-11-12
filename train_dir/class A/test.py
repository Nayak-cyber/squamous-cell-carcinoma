# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import tensorflow as tf


# # Load VGG16 pre-trained model, excluding the top layer
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# # Unfreeze the last few layers in the base model for fine-tuning
# for layer in base_model.layers[-4:]:
#     layer.trainable = True

# # Add custom layers for SCC classification
# x = Flatten()(base_model.output)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# output = Dense(1, activation='sigmoid')(x)

# model = Model(inputs=base_model.input, outputs=output)

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# # Data generator with validation split
# train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

# # Training generator
# train_generator = train_datagen.flow_from_directory(
#     'data/train',
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='binary',
#     subset='training'  # Use this for training subset
# )

# # Validation generator
# val_generator = train_datagen.flow_from_directory(
#     'data/train',
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation'  # Use this for validation subset
# )

# # Train the model with validation data
# model.fit(train_generator, epochs=5, validation_data=val_generator)
# model.save('scc_detection_model.keras')
# print("Model training complete and saved as 'scc_detection_model.keras'")

# # Detection function
# def detect_scc(image_path, model_path='scc_detection_model.keras'):
#     model = tf.keras.models.load_model(model_path)
    
#     # Load and preprocess the image
#     img = load_img(image_path, target_size=(128, 128))
#     img_array = img_to_array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Make prediction
#     prediction = model.predict(img_array)[0][0]
#     return "SCC Detected" if prediction > 0.46 else f"No SCC Detected {prediction}"

# # Example usage
# result = detect_scc('C:\\Users\\gunja\\OneDrive\\CODING\\Dinesh Project\\detect\\1.jpeg')
# print("Detection result:", result)
# result1 = detect_scc('C:\\Users\\gunja\\OneDrive\\CODING\\Dinesh Project\\detect\\scarlet.jpeg')
# print("Detection result:", result1)
# result2 = detect_scc('C:\\Users\\gunja\\OneDrive\\CODING\\Dinesh Project\\detect\\5.jpeg')
# print("Detection result:", result2)
# result3 = detect_scc('C:\\Users\\gunja\\OneDrive\\CODING\\Dinesh Project\\detect\\3.jpg')
# print("Detection result:", result3)
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt

# Set data directories
train_dir = r'C:\Users\samin\deepesh mini\gaussian_filtered_images\gaussian_filtered_images'
val_dir = r'C:\Users\samin\deepesh mini\gaussian_filtered_images\gaussian_filtered_images'

# Data Augmentation
datagen = ImageDataGenerator(rescale=1./255,
                             zoom_range=0.2,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             validation_split=0.2)

# Prepare training data
train_data = datagen.flow_from_directory(train_dir,
                                         target_size=(224, 224),
                                         batch_size=32,
                                         class_mode='categorical',
                                         subset='training')

# Prepare validation data
valid_data = datagen.flow_from_directory(val_dir,
                                         target_size=(224, 224),
                                         batch_size=32,
                                         class_mode='categorical',
                                         subset='validation')

# Learning rate schedule
def lr_rate(epoch, lr):
    if epoch < 10:
        return 0.0001
    elif epoch <= 15:
        return 0.0005
    elif epoch <= 30:
        return 0.0001
    else:
        return lr * (epoch / (1 + epoch))

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_rate)

# Build the model using EfficientNetB0
base_model = efn.EfficientNetB0(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
base_model.trainable = False  # Freeze the base model

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=valid_data, callbacks=[lr_callback], epochs=40, verbose=1)

# Plotting training curves
def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:  # Set up the subplots on the first call
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('Model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Valid'])

# Display loss and accuracy curves
display_training_curves(history.history['loss'], history.history['val_loss'], 'Loss', 211)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy', 212)

plt.show()