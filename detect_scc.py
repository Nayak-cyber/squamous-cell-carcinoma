import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def create_and_train_model():
    # Load VGG16 pre-trained model, excluding the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-8:]:
        layer.trainable = True

    # Add custom layers for SCC classification
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model with a learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Data generator with validation split and augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    # Train the model
    model.fit(train_generator, epochs=5, validation_data=val_generator)
    model.save('scc_detection_model.keras')
    print("Model training complete and saved as 'scc_detection_model.keras'")

# Call the function to train and save the model
create_and_train_model()

# Detection function for inference
def detect_scc(image_path, model_path='scc_detection_model.keras', threshold=0.7):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    return f"SCC Detected {prediction}" if prediction > threshold else f"No SCC Detected {prediction}"

# Example usage for detection
if __name__ == "__main__":
    result = detect_scc('C:\\Users\\gunja\\OneDrive\\CODING\\Dinesh Project\\detect\\1.jpeg')
    print("Detection result:", result)
