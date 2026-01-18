import streamlit as st
import pickle

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App settings
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# 🖼️ Add an image or logo
st.image("https://cdn-icons-png.flaticon.com/512/2921/2921222.png", width=120)
# You can replace the URL with your own image file, e.g. "logo.png" if it’s in the same folder

# App title
st.title("📰 Fake News Detection App")
st.markdown("Enter any news article or headline below to check if it’s **Fake** or **Real**.")

# Text input
news_text = st.text_area("🗞️ Paste your news content here:", height=200, placeholder="Type or paste the news text here...")

# Predict button
if st.button("🔍 Check News Authenticity"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        # Transform input text
        transformed_text = vectorizer.transform([news_text])

        # Predict using trained model
        prediction = model.predict(transformed_text)[0]

        # Decode prediction
        if prediction == 1:
            st.success("✅ This news seems **REAL**.")
        else:
            st.error("🚨 This news seems **FAKE**.")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and scikit-learn")
