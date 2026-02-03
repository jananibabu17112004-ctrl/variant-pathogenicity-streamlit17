import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# ----------------------------
# Load model and encoders
# ----------------------------
MODEL_PATH = "transssv_style_model.keras"
ENCODER_PATH = "encoders.pkl"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

gene_le = encoders["GeneSymbol"]
type_le = encoders["Type"]
review_le = encoders["ReviewStatus"]
assembly_le = encoders["Assembly"]

# ----------------------------
# Helper functions
# ----------------------------
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return 0  # fallback for unseen category

def predict_variant(gene, variant_type, review_status, assembly):
    input_data = np.array([[
        safe_encode(gene_le, gene),
        safe_encode(type_le, variant_type),
        safe_encode(review_le, review_status),
        safe_encode(assembly_le, assembly)
    ]])

    preds = model.predict(input_data, verbose=0)[0]
    classes = ["BENIGN", "VUS", "PATHOGENIC"]

    return classes[np.argmax(preds)], preds

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Genetic Variant Classifier", layout="centered")

st.title("🧬 Genetic Variant Classification")
st.write("Predict whether a genetic variant is **Benign, VUS, or Pathogenic**")

gene = st.selectbox("Gene Symbol", gene_le.classes_)
variant_type = st.selectbox("Variant Type", type_le.classes_)
review_status = st.selectbox("Review Status", review_le.classes_)
assembly = st.selectbox("Genome Assembly", assembly_le.classes_)

if st.button("🔍 Predict"):
    result, probs = predict_variant(
        gene, variant_type, review_status, assembly
    )

    st.subheader(f"🧾 Prediction: **{result}**")

    # Probability bar chart
    fig, ax = plt.subplots()
    ax.bar(["BENIGN", "VUS", "PATHOGENIC"], probs)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    st.pyplot(fig)
    model = tf.keras.models.load_model(
    "transssv_style_model.keras",
    compile=False
)

