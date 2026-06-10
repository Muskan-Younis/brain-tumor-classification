import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model("brain_tumor_model.h5")

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = 224

# CHANGE THIS TO YOUR IMAGE
img_path = "test_mri.jpg"

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# -----------------------------
# PREDICTION
# -----------------------------
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

print("Predicted Class:", predicted_class)

# -----------------------------
# GET LAST CONVOLUTION LAYER
# -----------------------------
last_conv_layer = model.get_layer("Conv_1")

# -----------------------------
# CREATE GRAD-CAM MODEL
# -----------------------------
grad_model = tf.keras.models.Model(
    [model.inputs],
    [last_conv_layer.output, model.output]
)

# -----------------------------
# COMPUTE GRADIENTS
# -----------------------------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, predicted_class]

# gradients
grads = tape.gradient(loss, conv_outputs)

# mean intensity of gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# feature maps
conv_outputs = conv_outputs[0]

# multiply feature maps by gradients
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# -----------------------------
# DISPLAY HEATMAP
# -----------------------------
plt.matshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.show()

# -----------------------------
# OVERLAY HEATMAP ON IMAGE
# -----------------------------
img = cv2.imread(img_path)

heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img

cv2.imwrite("gradcam_output.jpg", superimposed_img)

print("Grad-CAM saved as gradcam_output.jpg")

-----------------------------
TRAIN BUTTON
-----------------------------
if st.button("🚀 Train Model"):

    model = build_model(train_data.num_classes)

    progress = st.progress(0)
    status = st.empty()

    history = {"accuracy": [], "val_accuracy": []}

    for epoch in range(EPOCHS):
        status.text(f"Training... Epoch {epoch+1}/{EPOCHS}")

        hist = model.fit(train_data, validation_data=test_data, epochs=1, verbose=0)

        history["accuracy"].append(hist.history["accuracy"][0])
        history["val_accuracy"].append(hist.history["val_accuracy"][0])

        progress.progress((epoch+1)/EPOCHS)

    st.success("✅ Training Complete!")

    -----------------------------
    EVALUATION
    -----------------------------
    loss, accuracy = model.evaluate(test_data, verbose=0)
    st.subheader(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_data.classes

    class_labels = list(test_data.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred_classes)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -----------------------------
    # CLASSIFICATION REPORT
    # -----------------------------
    report = classification_report(y_true, y_pred_classes, target_names=class_labels)
    st.text("📊 Classification Report")
    st.text(report)

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    model.save("brain_tumor_model.h5")
    st.success("💾 Model saved as brain_tumor_model.h5")

    st.session_state["model"] = model

