import gradio as gr
import numpy as np
import cv2
import pickle
from PIL import Image

# ===============================
# LOAD MODEL & LABEL ENCODER
# ===============================
with open("final_dt_model_medical.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

CLASS_NAMES = list(label_encoder.classes_)

# ===============================
# IMAGE PREPROCESSING
# (MUST MATCH TRAINING PIPELINE)
# ===============================
def preprocess_image(image):
    if image is None:
        return None

    # PIL → NumPy
    image = np.array(image)

    # Convert RGB → GRAYSCALE
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to 64x64 (same as training)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)

    # Normalize
    image = image.astype("float32") / 255.0

    # Flatten → 4096 features
    image = image.flatten().reshape(1, -1)

    return image


# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_tumor(image):
    try:
        img = preprocess_image(image)

        if img is None:
            return "Please upload an image."

        probs = model.predict_proba(img)[0]
        max_prob = np.max(probs)
        pred_class = np.argmax(probs)

        # Entropy calculation
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        # Rejection logic (Decision Tree–safe)
        if max_prob < 0.25 or entropy > 1.2:
            return "⚠️ Uploaded image is NOT a valid Brain MRI scan."

        pred_label = label_encoder.inverse_transform([pred_class])[0]

        return (
            f"Predicted Brain Tumor Type: {pred_label}\n"
            f"Confidence: {max_prob:.2f}"
        )

    except Exception as e:
        return f"Prediction Error: {str(e)}"





# ===============================
# GRADIO APP
# ===============================
app = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil", label="Upload Brain MRI Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Multi-Class Brain Tumor MRI Classification",
    description=(
        "This application classifies brain MRI images into four categories: "
        "Glioma, Meningioma, Pituitary Tumor, or No Tumor using a Decision Tree model."
    )
)

# ===============================
# LAUNCH
# ===============================
if __name__ == "__main__":
    app.launch()
