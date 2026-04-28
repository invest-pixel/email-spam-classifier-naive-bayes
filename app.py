import re
import joblib
import streamlit as st

from naive_bayes_from_scratch import NaiveBayesSpamClassifier


MODEL_PATH = "models/naive_bayes_model.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Classifier")
st.caption(
    "Multinomial Naive Bayes from scratch. "
    "Không dùng model sklearn để predict, chỉ dùng xác suất và chút niềm tin vào toán."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Không tìm thấy file model. Hãy chắc chắn file nằm ở: "
        "`models/naive_bayes_model.pkl`"
    )
    st.stop()
except Exception as e:
    st.error("Không load được model `.pkl`.")
    st.code(str(e))
    st.stop()


st.markdown("### Nhập nội dung email / tin nhắn")

email_text = st.text_area(
    label="Nội dung cần phân loại",
    height=220,
    placeholder="Example: Congratulations! You won a free prize. Click here now!"
)

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("🔍 Phân loại", use_container_width=True)

with col2:
    clear_button = st.button("🧹 Xóa", use_container_width=True)

if clear_button:
    st.rerun()

if predict_button:
    if not email_text.strip():
        st.warning("Nhập nội dung trước đã. Model không phân loại được khoảng trống trong linh hồn mày đâu.")
    else:
        cleaned = clean_text(email_text)

        prediction = model.predict_one(cleaned)
        probabilities = model.predict_proba_one(cleaned)

        not_spam_prob = probabilities.get(0, 0)
        spam_prob = probabilities.get(1, 0)

        st.markdown("### Kết quả")

        if prediction == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")

        st.markdown("### Xác suất dự đoán")

        st.write(f"**Not Spam:** {not_spam_prob:.2%}")
        st.progress(float(not_spam_prob))

        st.write(f"**Spam:** {spam_prob:.2%}")
        st.progress(float(spam_prob))

        with st.expander("Text sau khi xử lý"):
            st.write(cleaned)

        with st.expander("Thông tin model"):
            st.write("Model: Multinomial Naive Bayes from scratch")
            st.write("Feature representation: Bag-of-Words")
            st.write("Labels:")
            st.write("- `0`: Not Spam")
            st.write("- `1`: Spam")