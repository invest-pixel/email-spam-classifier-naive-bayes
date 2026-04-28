import re
import joblib

from naive_bayes_from_scratch import NaiveBayesSpamClassifier


MODEL_PATH = "models/naive_bayes_model.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    model = joblib.load(MODEL_PATH)

    text = input("Enter email/message text: ")

    cleaned = clean_text(text)

    prediction = model.predict_one(cleaned)
    probabilities = model.predict_proba_one(cleaned)

    if prediction == 1:
        print("Prediction: Spam")
    else:
        print("Prediction: Not Spam")

    print(f"Not Spam probability: {probabilities.get(0, 0):.2%}")
    print(f"Spam probability: {probabilities.get(1, 0):.2%}")


if __name__ == "__main__":
    main()