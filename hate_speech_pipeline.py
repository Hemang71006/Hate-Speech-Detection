from __future__ import annotations
import os
import re
import argparse
from collections import Counter
import joblib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from imblearn.over_sampling import RandomOverSampler

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag


def ensure_nltk_data() -> None:
    """
    Make sure required NLTK data is downloaded.

    Simple explanation: some text-processing tools (tokenizer, lemmatizer,
    stopwords list) require extra files from NLTK. This function checks for
    those files and downloads any that are missing. It is safe to run many
    times.
    """
    nltk_packages = [
        "punkt",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "omw-1.4",
    ]
    for pkg in nltk_packages:
        try:
            # If package already exists this will succeed and do nothing
            nltk.data.find(pkg)
        except Exception:
            # Otherwise download the package once
            nltk.download(pkg)


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the CSV file into a pandas DataFrame.

    Simple explanation: If you pass a path, the function uses it. If not,
    it tries a few common paths where the Davidson CSV is usually stored.
    If no file is found it raises an error. Returns the loaded DataFrame.
    """
    if path and os.path.exists(path):
        csv_path = path
    else:
        # common places the dataset might be located
        candidates = [
            "dataset/data/labeled_data.csv",
            "dataset/labeled_data.csv",
            "data/labeled_data.csv",
            "labeled_data.csv",
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if not csv_path:
        # helpful error: tells user to provide correct path
        raise FileNotFoundError("Could not find labeled_data.csv — provide --data path")
    # read CSV into a DataFrame and return it
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    return df


def choose_text_label_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Pick which columns are the text and the label.

    Simple explanation: the function looks for common names like 'tweet' or
    'text' for the text column and 'class' or 'label' for the label column.
    If those names are not present it falls back to the first text-like
    column and a numeric column for labels. Returns (text_col, label_col).
    """
    # candidate names we expect in the Davidson dataset or similar
    text_candidates = [c for c in df.columns if c.lower() in ("tweet", "text", "message", "comment", "body")]
    label_candidates = [c for c in df.columns if c.lower() in ("class", "label", "category", "hate_category")]

    if text_candidates:
        text_col = text_candidates[0]
    else:
        # fallback: use first object (string) column
        text_col = [c for c in df.columns if df[c].dtype == "object"][0]

    if label_candidates:
        label_col = label_candidates[0]
    else:
        # fallback: prefer numeric columns for labels
        numeric_cols = [c for c in df.columns if str(df[c].dtype).startswith(("int", "float"))]
        if numeric_cols:
            label_col = numeric_cols[0]
        else:
            # last resort: use the last column
            label_col = df.columns[-1]

    return text_col, label_col


def analyze_distribution(df: pd.DataFrame, label_col: str, show_plot: bool = True) -> Counter:
    """
    Print how many examples are in each class and optionally plot a bar chart.

    Simple explanation: this helps you see if the dataset is imbalanced
    (one class has many more examples than others). Returns a Counter.
    """
    counts = Counter(df[label_col].dropna().values)
    print("Label counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if show_plot:
        # small bar chart for a quick visual check
        plt.figure(figsize=(6, 4))
        sns.barplot(x=list(map(str, counts.keys())), y=list(counts.values()))
        plt.title("Class distribution")
        plt.ylabel("Count")
        plt.xlabel("Label")
        plt.tight_layout()
        plt.show()
    return counts


def nltk_pos_to_wordnet(pt: str) -> str:
    """
    Map NLTK POS tags to WordNet POS tags.

    Simple explanation: lemmatizer needs WordNet-style tags (noun/verb/adj/adv).
    NLTK POS tags look different (e.g. 'NN', 'VB', 'JJ') so we translate them.
    """
    if pt.startswith("J"):
        return wordnet.ADJ
    if pt.startswith("V"):
        return wordnet.VERB
    if pt.startswith("N"):
        return wordnet.NOUN
    if pt.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_text(text: str) -> str:
    """
    Do simple text cleaning and return a plain lowercase string of words.

    Steps performed:
    - make lowercase
    - remove URLs (http or www)
    - remove mentions starting with @
    - remove the '#' character (keep the word)
    - remove punctuation and digits (keep letters and spaces)
    - collapse multiple spaces
    If input is not a string, returns empty string.
    """
    if not isinstance(text, str):
        return ""
    # lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r"http\S+|www\.[^\s]+", " ", text)
    # remove user mentions like @username
    text = re.sub(r"@\w+", " ", text)
    # replace '#' with space so hashtags become words
    text = re.sub(r"#", " ", text)
    # remove any non-letter characters (numbers, punctuation)
    text = re.sub(r"[^a-z\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str, stopwords_set: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Full preprocessing for a single text string.

    Simple explanation: calls `clean_text`, splits into words (tokenize),
    removes common stopwords (like 'the', 'and'), removes very short tokens,
    finds part-of-speech tags (so lemmatizer works better), and then
    lemmatizes each word (reduces 'running'->'run'). Returns a cleaned
    space-joined string of lemmas.
    """
    t = clean_text(text)
    if not t:
        return ""
    # split into tokens (words)
    toks = word_tokenize(t)
    # remove stopwords and tokens of length 1
    toks = [w for w in toks if w not in stopwords_set and len(w) > 1]
    # tag tokens with parts of speech. If the POS tagger resource is missing
    # (common NLTK lookup error), fall back to lemmatizing without POS.
    try:
        pos_tags = pos_tag(toks)
        # lemmatize each token with correct POS
        lemmas = [lemmatizer.lemmatize(w, nltk_pos_to_wordnet(p)) for w, p in pos_tags]
    except LookupError:
        # fallback: lemmatize using default POS (noun) if tagger is unavailable
        lemmas = [lemmatizer.lemmatize(w) for w in toks]
    return " ".join(lemmas)


def vectorize_text(corpus: pd.Series, max_features: int = 10000) -> tuple[TfidfVectorizer, np.ndarray]:
    """
    Fit a TF-IDF vectorizer on the preprocessed text and return (vectorizer, X).

    Simple explanation: transforms text into numerical features using
    TF-IDF. `max_features` limits vocabulary size for speed/memory.
    """
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = tfidf.fit_transform(corpus.fillna(""))
    return tfidf, X


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained model on test data and print results.

    Simple explanation: predict labels for `X_test`, print the standard
    classification report (precision, recall, f1 per class), draw a
    confusion matrix plot, and return a small dict with weighted
    accuracy/precision/recall/f1 values.
    """
    # predict
    y_pred = model.predict(X_test)
    # compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    # show classification report (per-class metrics)
    print(classification_report(y_test, y_pred, zero_division=0))
    # confusion matrix plot for visual debugging
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.show()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main(args: argparse.Namespace) -> None:
    """
    Run the full pipeline using command-line arguments in `args`.

    Simple walkthrough of steps performed inside main:
    1. Download small NLTK datasets if missing.
    2. Load CSV and choose which columns are text and label.
    3. Show class counts and optionally plot distribution.
    4. Preprocess each text into a cleaned, lemmatized string.
    5. Convert cleaned text into TF-IDF features.
    6. Split into train/test (80/20) using stratified split.
    7. Train Logistic Regression and Linear SVM using class weights.
    8. Evaluate both models (reports + confusion matrices).
    9. Try oversampling the training set, retrain a Logistic Regression,
       and evaluate it.
    10. Pick the best model (by weighted F1) and save model + vectorizer.
    """
    # Ensure NLTK data required for tokenization and lemmatization
    ensure_nltk_data()
    # Prepare stopwords and lemmatizer objects used in preprocessing
    stopwords_set = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Load dataset from CSV
    df = load_dataset(args.data)
    print("Dataset shape:", df.shape)

    # Decide which columns contain the text and labels
    text_col, label_col = choose_text_label_columns(df)
    print("Text column:", text_col, "Label column:", label_col)
    print(df[[text_col, label_col]].head(5))

    # Create a run-specific directory to save outputs for this run
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base_dir = Path("artifacts")
    run_label = args.run_name if getattr(args, 'run_name', None) else f"run_{ts}"
    run_dir = base_dir / run_label
    if run_dir.exists() and not getattr(args, 'force', False):
        # avoid overwriting an existing named run unless --force is provided
        run_dir = base_dir / f"{run_label}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # save a small reference of the raw dataset
    try:
        df.head(50).to_csv(run_dir / "dataset_head.csv", index=False)
    except Exception:
        pass

    # Show how classes are distributed (quick EDA) and save distribution
    counts = analyze_distribution(df, label_col, show_plot=args.plot)
    try:
        # counts may be Counter; convert to plain dict for JSON
        with open(run_dir / "class_distribution.json", "w", encoding="utf-8") as fh:
            json.dump(dict(counts), fh, ensure_ascii=False, indent=2)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=list(map(str, counts.keys())), y=list(counts.values()))
        plt.title("Class distribution")
        plt.ylabel("Count")
        plt.xlabel("Label")
        plt.tight_layout()
        plt.savefig(run_dir / "class_distribution.png")
        plt.close()
    except Exception:
        pass

    # Preprocess texts (may take time). If cleaned column already exists and
    # user did not pass --force, reuse it to avoid recomputing.
    print("Preprocessing texts (this may take a while)...")
    if "text_clean" in df.columns and not getattr(args, 'force', False):
        print("Reusing existing 'text_clean' column (use --force to recompute).")
    else:
        df["text_clean"] = df[text_col].astype(str).map(lambda t: preprocess_text(t, stopwords_set, lemmatizer))
    empty = (df["text_clean"].str.strip() == "").sum()
    print(f"Empty texts after cleaning: {empty} ({empty/len(df):.2%})")
    try:
        df[[text_col, "text_clean"]].head(200).to_csv(run_dir / "sample_cleaned_texts.csv", index=False)
    except Exception:
        pass

    # Convert text to TF-IDF features
    tfidf, X = vectorize_text(df["text_clean"], max_features=args.max_features)
    y = df[label_col].values
    try:
        vocab = {w: i for w, i in tfidf.vocabulary_.items()}
        with open(run_dir / "tfidf_vocab.json", "w", encoding="utf-8") as fh:
            json.dump(vocab, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Train/test split (stratified to keep label proportions)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Train/Test shapes:", X_train.shape, X_test.shape)

    # Train models using class weights to partially address imbalance
    print("Training Logistic Regression (class_weight='balanced')...")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)

    print("Training LinearSVC (class_weight='balanced')...")
    svm = LinearSVC(class_weight="balanced", max_iter=20000, random_state=42)
    svm.fit(X_train, y_train)

    # Evaluate both trained models on the held-out test set and save reports
    results = {}

    print("Evaluating Logistic Regression")
    res_lr = evaluate_model(lr, X_test, y_test)
    results["LogisticRegression"] = res_lr
    try:
        with open(run_dir / "classification_report_LogisticRegression.txt", "w", encoding="utf-8") as fh:
            fh.write(classification_report(y_test, lr.predict(X_test), zero_division=0))
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, lr.predict(X_test)), annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - LogisticRegression")
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()
        plt.savefig(run_dir / "confusion_LogisticRegression.png")
        plt.close()
    except Exception:
        pass

    print("Evaluating LinearSVC")
    res_svm = evaluate_model(svm, X_test, y_test)
    results["LinearSVC"] = res_svm
    try:
        with open(run_dir / "classification_report_LinearSVC.txt", "w", encoding="utf-8") as fh:
            fh.write(classification_report(y_test, svm.predict(X_test), zero_division=0))
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, svm.predict(X_test)), annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - LinearSVC")
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()
        plt.savefig(run_dir / "confusion_LinearSVC.png")
        plt.close()
    except Exception:
        pass

    # Try an oversampling approach: duplicate minority examples in train set
    print("Handling imbalance via RandomOverSampler on training set...")
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    print("Resampled distribution:", Counter(y_res))
    lr_ros = LogisticRegression(max_iter=2000, random_state=42)
    lr_ros.fit(X_res, y_res)
    res_ros = evaluate_model(lr_ros, X_test, y_test)
    results["LR_oversampled"] = res_ros
    try:
        with open(run_dir / "classification_report_LR_oversampled.txt", "w", encoding="utf-8") as fh:
            fh.write(classification_report(y_test, lr_ros.predict(X_test), zero_division=0))
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, lr_ros.predict(X_test)), annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - LR_oversampled")
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()
        plt.savefig(run_dir / "confusion_LR_oversampled.png")
        plt.close()
    except Exception:
        pass

    # Choose best model based on weighted F1 score and save it
    best_name, best_score = None, -1
    for name, met in results.items():
        print(f"{name}: F1={met['f1']:.4f} Acc={met['accuracy']:.4f}")
        if met["f1"] > best_score:
            best_score = met["f1"]
            best_name = name
    print(f"Selected best model: {best_name} (F1={best_score:.4f})")

    model_map = {"LogisticRegression": lr, "LinearSVC": svm, "LR_oversampled": lr_ros}
    best_model = model_map[best_name]

    # Save summary metrics and artifacts into run directory
    try:
        with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    joblib.dump(best_model, run_dir / "best_model.joblib")
    joblib.dump(tfidf, run_dir / "tfidf_vectorizer.joblib")
    print(f"Saved artifacts to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hate Speech Detection pipeline")
    parser.add_argument("--data", type=str, default="dataset/data/labeled_data.csv", help="Path to labeled_data.csv")
    parser.add_argument("--max-features", dest="max_features", type=int, default=10000)
    parser.add_argument("--plot", action="store_true", help="Show class distribution plot")
    parser.add_argument("--run-name", dest="run_name", type=str, default=None, help="Optional name for this run (used to name the output folder)")
    parser.add_argument("--force", action="store_true", help="Force recomputation (don't reuse existing preprocessed/text_clean or run folder)")
    args = parser.parse_args()
    main(args)
