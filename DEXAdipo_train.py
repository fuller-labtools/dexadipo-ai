import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

def display_augmented_images(datagen, images, labels, num_images=5):
    fig, axs = plt.subplots(1, num_images, figsize=(20, 20))
    for imgs, lbls in datagen.flow(images, labels, batch_size=num_images):
        for i in range(num_images):
            axs[i].imshow(imgs[i].reshape(new_img_size), cmap='gray')
            axs[i].axis('off')
        break  # Only show one batch
    plt.show()

# Load CSV
label_df = pd.read_csv('C:/Users/DOC/Documents/Oli/Python/DXAPREDACU_PNG_SUBQ.csv')

def regression_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Initial Conv layer
    x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Deeper Residual Blocks
    for filters in [64, 128, 256, 512]:
        for _ in range(1):  # One residual block for each filter size
            # Shortcut
            shortcut = x
            
            # First convolution
            x = layers.Conv2D(filters, 3, activation=None, padding='same', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            # Second convolution
            x = layers.Conv2D(filters, 3, activation=None, padding='same', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            
            # Adding shortcut to the output (Residual Connection)
            x = layers.Add()([x, layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut)])
            x = layers.ReLU()(x)
        
        # MaxPooling
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully Connected Layers
    x = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x) 
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

new_img_size = (256, 256)

# Model creation
model = regression_model((new_img_size[0], new_img_size[1], 1))

def load_and_preprocess(image_path, label, img_size=new_img_size):
    # Load PNG file
    img = imread(image_path, as_gray=True)
    # Resize image
    img = resize(img, img_size, mode='reflect', anti_aliasing=True)
    # Scale image to range [0, 1]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    return img, label

# Load Data
data = []
labels = []
for idx, row in label_df.iterrows():
    img_path = os.path.join('C:/Users/DOC/Documents/Oli/Data/DXAPREDACU_PNG/', row['FILENAMES'])
    if os.path.exists(img_path) and img_path.endswith('.png'):
        img, label = load_and_preprocess(img_path, row['SUBQ'])
        data.append(img)
        labels.append(label)

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split Data
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Custom R-square metric
class RSquare(tf.keras.metrics.Metric):
    def __init__(self, name="r_square", **kwargs):
        super(RSquare, self).__init__(name=name, **kwargs)
        self.sse = self.add_weight(name='sse', initializer='zeros')
        self.sst = self.add_weight(name='sst', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error = tf.reduce_sum(tf.square(y_true - y_pred))
        mean_true = tf.reduce_mean(y_true)
        total = tf.reduce_sum(tf.square(y_true - mean_true))
        self.sse.assign_add(error)
        self.sst.assign_add(total)

    def result(self):
        return 1 - (self.sse / self.sst)

    def reset_state(self):
        self.sse.assign(0.0)
        self.sst.assign(0.0)

# Custom combined loss function for optimization
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    r_square = 1 - (tf.reduce_sum(tf.square(y_true - y_pred)) /
                    tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))))
    combined = 0.6 * mse + 0.8 * (1 - r_square)  # Adjust weights as necessary
    return combined

# Compile the model with the custom combined loss function and additional metrics
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=combined_loss, metrics=['mse', RSquare()])

models_dir = 'C:/Users/DOC/Documents/Oli/Python/saved_models'
os.makedirs(models_dir, exist_ok=True)

checkpoint_path = os.path.join(models_dir, 'best_initial_model.h5')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

# Add ReduceLROnPlateau callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, 
                                                 patience=800, min_delta=0.001, 
                                                 min_lr=0.000001, verbose=1)

# Reduced Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=5, width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=False, vertical_flip=False)
datagen.fit(X_train)

# Display a few images before training
display_augmented_images(datagen, X_train, y_train, num_images=20)

# Custom callback to log validation R-squared and MSE
class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        val_predictions = self.model.predict(X_val).squeeze()
        mse = mean_squared_error(y_val, val_predictions)
        r_square = r2_score(y_val, val_predictions)
        logs['val_mse'] = mse
        logs['val_r_square'] = r_square
        print(f'Epoch {epoch+1}: val_mse: {mse}, val_r_square: {r_square}')

validation_metrics_callback = ValidationMetricsCallback(validation_data=(X_val, y_val))

# -----------------------------
# TRAINING TIME MEASUREMENT
# -----------------------------
total_train_start = time.perf_counter()

# Initial Training with Data Augmentation and ReduceLROnPlateau Callback
phase1_start = time.perf_counter()
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=8000,
    callbacks=[model_checkpoint_callback, reduce_lr, validation_metrics_callback]
)
phase1_end = time.perf_counter()

# Load the best model from the initial training phase
best_model_path_initial = os.path.join(models_dir, 'best_initial_model.h5')
best_model = tf.keras.models.load_model(
    best_model_path_initial,
    custom_objects={"RSquare": RSquare, "combined_loss": combined_loss}
)

# Recompile the best model with a lower learning rate for the second training phase
best_model.compile(optimizer=tf.keras.optimizers.Adam(1e-6), loss=combined_loss, metrics=['mse', RSquare()])

# Second phase training (saving model each epoch)
phase2_start = time.perf_counter()
for epoch in range(800): 
    second_checkpoint_path = os.path.join(models_dir, f'second_phase_model_epoch_{epoch}.h5')
    second_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=second_checkpoint_path, save_best_only=False)

    # Train for one epoch
    best_model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=1,
        callbacks=[second_model_checkpoint_callback, validation_metrics_callback]
    )
phase2_end = time.perf_counter()

total_train_end = time.perf_counter()

phase1_time_sec = phase1_end - phase1_start
phase2_time_sec = phase2_end - phase2_start
total_train_time_sec = total_train_end - total_train_start

print(f"Phase 1 training time: {phase1_time_sec/60:.2f} minutes")
print(f"Phase 2 training time: {phase2_time_sec/60:.2f} minutes")
print(f"Total training time (both phases): {total_train_time_sec/3600:.2f} hours")

# -----------------------------
# MODEL SELECTION & EVALUATION
# -----------------------------
best_combined_score = -float('inf')  # Initialize with a very low score
best_model_path = ''

def calculate_r2(y_true, y_pred):
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / ss_tot)

for epoch in range(800):  # Assuming you have 800 models to evaluate
    model_path = os.path.join(models_dir, f'second_phase_model_epoch_{epoch}.h5')
    current_model = tf.keras.models.load_model(
        model_path, custom_objects={"RSquare": RSquare, "combined_loss": combined_loss}
    )

    preds = current_model.predict(X_temp).squeeze()
    mse = mean_squared_error(y_temp, preds)
    r_square = calculate_r2(y_temp, preds)

    # Normalize MSE to a 0-1 scale
    normalized_mse_score = 1 / (1 + mse)

    # Combine R-squared and normalized MSE (you can adjust the weights)    
    combined_score = 0.8 * r_square + 0.6 * normalized_mse_score

    # Select model based on the combined score
    if combined_score > best_combined_score:
        best_combined_score = combined_score
        best_model_path = model_path

# Load the best model and report
best_model = tf.keras.models.load_model(
    best_model_path, custom_objects={"RSquare": RSquare, "combined_loss": combined_loss}
)
print(f"Best Model: {best_model_path} - Combined Score: {best_combined_score}, MSE: {mse}, R-square: {r_square}")

# Predict on the test data with the best model
predictions = best_model.predict(X_temp).squeeze()

# Print the evaluation metrics for the best model
mse = mean_squared_error(y_temp, predictions)
r_square = calculate_r2(y_temp, predictions)
print(f"Best Model MSE on Test Data: {mse}")
print(f"Best Model R-squared on Test Data: {r_square}")

# Compare predictions with actual values for the best model
comparison_df = pd.DataFrame({'Actual': y_temp, 'Predicted': predictions})
print(comparison_df)

# Save the best model
best_model.save(os.path.join(models_dir, 'final_best_model.h5'))
