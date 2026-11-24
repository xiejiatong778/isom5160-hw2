"""
ISOM5160 Homework 2 — SMS Spam / Smishing Alert App
Course: ISOM5160-L3 Python Programming for Business Applications, Fall 2026 (PG ICSM)

INSTRUCTIONS (read carefully)
- Complete the marked STUDENT sections and the function bodies below.
- Keep these function names: load_model, load_test_samples, classify_text,
  compute_accuracy, main.
- Call main() at the bottom of this file (the starter already does this).
- In GitHub and on Streamlit Cloud, this file must be named app.py.
- Submit to Canvas: a zip file containing app.py, requirements.txt, and
  Homework2_<StudentID>_link.txt (Streamlit Cloud URL and GitHub URL).
  Put those three files at the root of the zip, not inside a subfolder.
- Do NOT fine-tune a model. Do NOT scrape the web.
- Do NOT change DATASET_ID, MODEL_ID, N_TEST, or SPLIT_SEED.
- When you deploy, open Advanced settings and set Python to 3.12.
  If the app is already deployed, delete it and redeploy — the Python
  version cannot be changed in place.
- If the app calls a Hugging Face model, store a fine-grained Hugging Face
  API token in Streamlit Cloud secrets as HF_TOKEN. Do not put the token
  in app.py, requirements.txt, or any public file.
- First run may download the model and dataset (needs network). CPU is OK.

SECRET PHRASE (must appear on the app page): ISOM5160-HW2-SMS
"""

# =============================================================================
# STUDENT ANSWERS BLOCK (fill in; keep this in the SAME .py file)
# =============================================================================
# Student ID: 21397884
#
# GenAI acknowledgment (required if you used GenAI; otherwise write "None"):
#   - Tools used: Claude Code (Anthropic)
#   - How you used them: Claude Code wrote the code for classify_text,
#     compute_accuracy and the marked parts of main(), added the retry
#     around the transformers import, tested the app locally, pushed the
#     files to GitHub and helped debug the Streamlit Cloud deployment.
#     The Q1 and Q2 answers below are my own and were not written by GenAI.
#
# Q1. Who is the primary business user of this app?
#     (one short sentence; e.g., an employee or a security analyst reviewing
#      inbound SMS for spam or scam-like messages)
#   YOUR_ANSWER: A fraud-operations officer at a mobile carrier who
#   decides which incoming SMS should be flagged as suspicious before they
#   reach subscribers.
#
# Q2. Look at the accuracy from your 100-sample test. Is accuracy enough
#     for a business to rely on this app, and why or why not?
#     2–4 short sentences max. You MUST cite the accuracy figure from
#     YOUR app run. Comment on the cost of a false positive (blocking a
#     legitimate SMS, such as a bank or one-time code message) versus a
#     false negative (missing a spam or scam-like message).
#   YOUR_ANSWER: Not on its own: my run reached 99.00% accuracy, with 0
#   false positives and 1 false negative. But 88 of the 100 messages were
#   legitimate, so labelling everything "legitimate" would still score
#   88%; the more useful figure is that the app caught 11 of the 12 spam
#   messages. A missed scam can cost a subscriber money, while a wrongly
#   blocked bank alert or one-time code stops a real transaction, and 100
#   samples are too few to trust a zero false-positive rate. I would use
#   the app as a first-pass warning with human review, not as an automatic
#   blocker.
#
# =============================================================================

import streamlit as st
from datasets import load_dataset

try:
    from transformers import pipeline
except ImportError:
    # On a Streamlit Cloud cold start two script runs can import transformers
    # at the same moment and one of them fails. Wait briefly and try again.
    import time

    time.sleep(3)
    from transformers import pipeline

# Locked by the instructor — do not change these four values.
DATASET_ID = "ucirvine/sms_spam"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
N_TEST = 100
SPLIT_SEED = 42

# Course identity (required)
STUDENT_ID = "21397884"  # STUDENT: replace with your real student ID
COURSE_SECRET = "ISOM5160-HW2-SMS"  # do not change


def map_label(raw_label):
    """Map model or dataset labels to the two display names used in this app."""
    text = str(raw_label).strip().lower()
    if text in {"ham", "legitimate", "label_0", "0"}:
        return "legitimate"
    if text in {"spam", "smishing", "label_1", "1"}:
        return "spam"
    return str(raw_label)


@st.cache_resource
def load_model(token):
    """Load the SMS text-classification model once and reuse it."""
    # This function is provided. Keep device="cpu" and
    # model_kwargs={"low_cpu_mem_usage": False} so Streamlit Cloud can load
    # the model on CPU.
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=token,
        device="cpu",
        truncation=True,
        max_length=512,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier


@st.cache_data
def load_test_samples():
    """
    Load the Hugging Face dataset and take a fixed set of 100 samples.

    The Hub copy has only a train split, so this homework holds out 100
    rows after a shuffle with SPLIT_SEED. Do not change the seed or N_TEST,
    so every student is tested on the same 100 messages.
    """
    raw = load_dataset(DATASET_ID, split="train")
    picked = raw.shuffle(seed=SPLIT_SEED).select(range(N_TEST))
    texts = list(picked["sms"])
    true_labels = [map_label(label) for label in picked["label"]]
    return texts, true_labels


def classify_text(classifier, text):
    """Return (display_label, confidence) for one SMS string."""
    # STUDENT: call the pipeline on this one text, map the raw label with
    # map_label(...), and return the display label and the confidence score.
    result = classifier(text)[0]  # e.g. {"label": "LABEL_1", "score": 0.98}
    display_label = map_label(result["label"])
    score = float(result["score"])
    return display_label, score


def compute_accuracy(pred_labels, true_labels):
    """Return accuracy as a float between 0 and 1."""
    # STUDENT: accuracy = (number of matches) / (number of samples).
    if len(true_labels) == 0:
        return 0.0
    matches = 0
    for pred, true in zip(pred_labels, true_labels):
        if pred == true:
            matches += 1
    accuracy = matches / len(true_labels)
    return accuracy


def main():
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    st.info(f"Student ID: {STUDENT_ID}  |  Secret: {COURSE_SECRET}")

    # Use a Hugging Face fine-grained token from Streamlit Cloud secrets
    # if one is set. Do not put the token in this file.
    hf_token = None
    if "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]

    classifier = load_model(hf_token)

    alert_text = st.text_area(
        "SMS message",
        value=(
            "WINNER!! You have been selected to receive a $900 prize. "
            "Claim now at http://prize-alert.example or the offer expires."
        ),
        height=140,
    )

    if st.button("Classify SMS"):
        if alert_text.strip() == "":
            st.warning("Please paste an SMS message first.")
        else:
            label, score = classify_text(classifier, alert_text)
            if label == "spam":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")
            st.write(f"Confidence: {score:.2%}")

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        with st.spinner("Loading samples and classifying 100 messages..."):
            texts, true_labels = load_test_samples()
            pred_labels = []
            rows = []
            for i, (text, true_label) in enumerate(zip(texts, true_labels), start=1):
                pred_label, score = classify_text(classifier, text)
                pred_labels.append(pred_label)
                short_text = text.strip()
                if len(short_text) > 60:
                    short_text = short_text[:60] + "..."
                rows.append(
                    {
                        "#": i,
                        "Text (short)": short_text,
                        "Ground truth": true_label,
                        "Prediction": pred_label,
                        "Confidence": f"{score:.2%}",
                        "Match": "Yes" if pred_label == true_label else "No",
                    }
                )
            accuracy = compute_accuracy(pred_labels, true_labels)

        correct = sum(1 for row in rows if row["Match"] == "Yes")
        st.metric("Accuracy", f"{accuracy:.2%}")
        st.write(f"{correct} of {len(rows)} messages classified correctly.")
        st.dataframe(rows, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
