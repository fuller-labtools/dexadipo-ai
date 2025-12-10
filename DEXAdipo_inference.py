import os
import pandas as pd
import tensorflow as tf
from skimage.transform import resize
import numpy as np
from skimage.io import imread

# ------------------------------------------------
# CONFIG: REPLACE THESE PLACEHOLDERS WITH YOUR PATHS
# ------------------------------------------------
MODEL_PATH = "ADD_PATH_TO_TRAINED_MODEL_H5"  # e.g., "/path/to/DXA_PRED_SUBQ_PNG_r2_99.h5"
NEW_DATASET_CSV = "ADD_PATH_TO_NEW_DATASET_CSV"  # CSV containing a 'FILENAMES' column
BASE_DIR = "ADD_PATH_TO_IMAGE_BASE_DIRECTORY"  # directory that contains the PNG files listed in CSV
OUTPUT_CSV = "ADD_PATH_TO_OUTPUT_PREDICTIONS_CSV"  # where to save outputs

# ------------------------------------------------
# Custom R-square metric
# ------------------------------------------------
class RSquare(tf.keras.metrics.Metric):
    def __init__(self, name="r_square", **kwargs):
        super(RSquare, self).__init__(name=name, **kwargs)
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.sst = self.add_weight(name="sst", initializer="zeros")

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

# ------------------------------------------------
# Custom combined loss function for optimization
# ------------------------------------------------
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    r_square = 1 - (
        tf.reduce_sum(tf.square(y_true - y_pred)) /
        tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    )
    combined = 0.6 * mse + 0.8 * (1 - r_square)  # Adjust weights as necessary
    return combined

# ------------------------------------------------
# Load the trained model
# ------------------------------------------------
best_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"RSquare": RSquare, "combined_loss": combined_loss},
    compile=False
)

# ------------------------------------------------
# Load CSV with image filenames
# ------------------------------------------------
new_image_df = pd.read_csv(NEW_DATASET_CSV)

new_img_size = (512, 512)  # Assuming the image size is the same

def load_and_preprocess(image_path, img_size=new_img_size):
    # Load PNG file
    img = imread(image_path, as_gray=True)
    # Resize image
    img = resize(img, img_size, mode="reflect", anti_aliasing=True)

    # Scale image to range [0, 1] with safe guard against zero range
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    return img

# ------------------------------------------------
# Load New Data
# ------------------------------------------------
new_data = []
matched_filenames = []

if "FILENAMES" not in new_image_df.columns:
    raise KeyError(
        "Your CSV must contain a 'FILENAMES' column with image file names."
    )

for _, row in new_image_df.iterrows():
    img_path = os.path.join(BASE_DIR, str(row["FILENAMES"]))
    if os.path.exists(img_path):
        img = load_and_preprocess(img_path)
        new_data.append(img)
        matched_filenames.append(row["FILENAMES"])

new_data = np.array(new_data)

if len(new_data) == 0:
    raise FileNotFoundError(
        "No images were found. Check:\n"
        "1) BASE_DIR (ADD_PATH_TO_IMAGE_BASE_DIRECTORY)\n"
        "2) The 'FILENAMES' values in your CSV\n"
        "3) That the files exist and are readable."
    )

# ------------------------------------------------
# Predict
# ------------------------------------------------
new_predictions = best_model.predict(new_data).squeeze()

# ------------------------------------------------
# Save + print results
# ------------------------------------------------
predictions_df = pd.DataFrame({
    "Filename": matched_filenames,
    "Predicted": new_predictions
})

pd.set_option("display.max_rows", None)
print(predictions_df)

predictions_df.to_csv(OUTPUT_CSV, index=False)
