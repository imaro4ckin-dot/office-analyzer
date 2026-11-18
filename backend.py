"""
backend.py — Who Said It? The Office NLP Classifier
====================================================
Pure backend: data loading, preprocessing, training, evaluation, and inference.
No UI dependencies (no Streamlit, no Plotly).

Usage
-----
Train and save all artifacts:
    python backend.py

Import in any UI layer:
    from backend import load_artifacts, predict

Artifacts written to models/:
    tfidf_vectorizer.joblib
    label_encoder.joblib
    logistic_regression.joblib
    xgboost_model.joblib
    kmeans_model.joblib
    pca_50.joblib
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display required
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix,
    silhouette_score, roc_auc_score, f1_score,
    accuracy_score, roc_curve, auc,
)
import xgboost as xgb
import joblib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_N_CHARACTERS = 7
N_CLUSTERS       = 5
TFIDF_MAX_FEATS  = 5000
PCA_COMPONENTS   = 50
MODELS_DIR       = "models"

VIBE_LABELS: dict[int, str] = {
    0: "Corporate Speak",
    1: "Sarcastic Banter",
    2: "Romantic / Emotional",
    3: "Dwight Energy",
    4: "Office Small Talk",
}

# ---------------------------------------------------------------------------
# NLTK bootstrap (runs once, silently)
# ---------------------------------------------------------------------------

def _download_nltk() -> None:
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(resource, quiet=True)

_download_nltk()

_STOP_WORDS  = set(stopwords.words("english"))
_lemmatizer  = WordNetLemmatizer()
_lemmatizer.lemmatize("warm")   # force-load the WordNet corpus now


# ===========================================================================
# Section 1 — Problem Formulation & EDA
# ===========================================================================

def load_data() -> pd.DataFrame:
    """Load The Office lines from HuggingFace and return a cleaned DataFrame."""
    dataset = load_dataset("jxm/the_office_lines")
    df = dataset["train"].to_pandas()

    # Defensive rename in case the dataset schema changes
    df = df.rename(columns={"character": "speaker", "text": "line_text"}, errors="ignore")
    assert "speaker"   in df.columns, f"Missing 'speaker' column.   Got: {df.columns.tolist()}"
    assert "line_text" in df.columns, f"Missing 'line_text' column.  Got: {df.columns.tolist()}"

    df = df.dropna(subset=["line_text", "speaker"])
    df = df[df["line_text"].str.strip().ne("")]
    df["speaker"] = df["speaker"].str.strip().str.title()

    print(f"Loaded {len(df)} rows, {df['speaker'].nunique()} unique speakers.")
    return df


def eda_class_imbalance(df: pd.DataFrame, save_path: str = "eda_line_counts.png") -> pd.Series:
    """
    Plot top-15 character line counts and save to disk.

    Returns
    -------
    pd.Series
        Full sorted value counts for all speakers.
    """
    line_counts = df["speaker"].value_counts()
    print(f"Total unique speakers: {len(line_counts)}")
    print("\nTop 15 by line count:")
    print(line_counts.head(15).to_string())

    fig, ax = plt.subplots(figsize=(13, 5))
    line_counts.head(15).plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Class Imbalance: Line Counts per Character (Top 15)", fontsize=14)
    ax.set_xlabel("Character")
    ax.set_ylabel("Number of Lines")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return line_counts


def filter_top_characters(df: pd.DataFrame, top_n: int = TOP_N_CHARACTERS) -> pd.DataFrame:
    """Keep only the top-N characters by line count."""
    top_chars = df["speaker"].value_counts().head(top_n).index.tolist()
    print(f"Modeling {top_n} characters: {top_chars}")
    df_f = df[df["speaker"].isin(top_chars)].copy().reset_index(drop=True)
    print(f"Rows after filtering: {len(df_f)}")
    print("\nFiltered class distribution:")
    print(df_f["speaker"].value_counts().to_string())
    return df_f


def preprocess(text: str) -> str:
    """Tokenize, lowercase, lemmatize, and remove stop words / non-alpha tokens."""
    tokens  = word_tokenize(text.lower())
    cleaned = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in _STOP_WORDS
    ]
    return " ".join(cleaned)


def build_tfidf(
    df: pd.DataFrame,
    max_features: int = TFIDF_MAX_FEATS,
) -> tuple[TfidfVectorizer, np.ndarray, LabelEncoder, np.ndarray]:
    """
    Preprocess text, fit TF-IDF, and encode labels.

    Returns
    -------
    tfidf : TfidfVectorizer
    X     : sparse matrix (n_samples, max_features)
    le    : LabelEncoder
    y     : ndarray of encoded labels
    """
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Preprocessing")
        df["clean_text"] = df["line_text"].progress_apply(preprocess)
    except ImportError:
        df["clean_text"] = df["line_text"].apply(preprocess)

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3)
    X     = tfidf.fit_transform(df["clean_text"])
    le    = LabelEncoder()
    y     = le.fit_transform(df["speaker"])

    print(f"TF-IDF feature matrix shape: {X.shape}")
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(le,    os.path.join(MODELS_DIR, "label_encoder.joblib"))
    print("Saved: tfidf_vectorizer.joblib, label_encoder.joblib")
    return tfidf, X, le, y


def print_top_tfidf_terms(
    tfidf: TfidfVectorizer,
    X: np.ndarray,
    le: LabelEncoder,
    y: np.ndarray,
    top_k: int = 10,
) -> None:
    """Print the top-K mean TF-IDF terms for each character (character signatures)."""
    feature_names = tfidf.get_feature_names_out()
    X_dense = X.toarray()
    print(f"Top {top_k} TF-IDF terms per character (character signatures):\n")
    for char_idx, char_name in enumerate(le.classes_):
        mask       = (y == char_idx)
        mean_tfidf = X_dense[mask].mean(axis=0)
        top_indices = mean_tfidf.argsort()[-top_k:][::-1]
        top_terms   = [feature_names[i] for i in top_indices]
        print(f"  {char_name:12s}: {top_terms}")


# ===========================================================================
# Section 2 — Unsupervised Clustering
# ===========================================================================

def reduce_pca(X_dense: np.ndarray, n_components: int = PCA_COMPONENTS) -> tuple[PCA, np.ndarray]:
    """Fit PCA for dimensionality reduction before K-Means."""
    pca   = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    print(f"Explained variance with {n_components} components: {pca.explained_variance_ratio_.sum():.2%}")
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_50.joblib"))
    print("Saved: pca_50.joblib")
    return pca, X_pca


def elbow_and_silhouette(
    X_pca: np.ndarray,
    k_range: range = range(2, 12),
    save_path: str = "clustering_elbow.png",
) -> tuple[list[float], list[float]]:
    """
    Compute inertia and silhouette score for each k and save elbow plot.

    Returns
    -------
    inertias    : list of inertia values
    silhouettes : list of silhouette scores
    """
    inertias, silhouettes = [], []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_pca, labels, sample_size=2000, random_state=42)
        silhouettes.append(sil)
        print(f"k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(list(k_range), inertias, "bo-", markersize=7)
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.3)
    axes[1].plot(list(k_range), silhouettes, "rs-", markersize=7)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(alpha=0.3)
    plt.suptitle("Choosing Optimal k for K-Means", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return inertias, silhouettes


def fit_kmeans(
    X_pca: np.ndarray,
    X_dense: np.ndarray,
    tfidf: TfidfVectorizer,
    n_clusters: int = N_CLUSTERS,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit final K-Means, print cluster term signatures, and save model.

    Returns
    -------
    km_final      : fitted KMeans
    cluster_labels: ndarray of cluster assignments
    """
    km_final       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_pca)

    feature_names = tfidf.get_feature_names_out()
    print(f"Top 8 TF-IDF terms per cluster (k={n_clusters}):\n")
    for c in range(n_clusters):
        mask         = (cluster_labels == c)
        cluster_mean = X_dense[mask].mean(axis=0)
        top_idx      = cluster_mean.argsort()[-8:][::-1]
        terms        = [feature_names[i] for i in top_idx]
        print(f"  Cluster {c} ({mask.sum():5d} lines): {terms}")

    joblib.dump(km_final, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    print("\nSaved: kmeans_model.joblib")
    return km_final, cluster_labels


def save_cluster_plot(
    X_pca50: np.ndarray,
    cluster_labels: np.ndarray,
    vibe_labels: dict[int, str] = VIBE_LABELS,
    save_path: str = "clusters_pca_static.png",
) -> None:
    """Project clusters to 2D via PCA and save a static scatter plot."""
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d   = pca_2d.fit_transform(X_pca50)
    vibes  = np.array([vibe_labels.get(c, f"Cluster {c}") for c in cluster_labels])

    sample_idx = np.random.default_rng(42).choice(len(X_2d), size=min(2500, len(X_2d)), replace=False)
    X_s, v_s   = X_2d[sample_idx], vibes[sample_idx]

    colors_map = {v: c for v, c in zip(np.unique(v_s),
                                        ["steelblue", "coral", "seagreen", "orchid", "goldenrod"])}
    fig, ax = plt.subplots(figsize=(10, 6))
    for vibe in np.unique(v_s):
        mask = (v_s == vibe)
        ax.scatter(X_s[mask, 0], X_s[mask, 1],
                   label=vibe, alpha=0.5, s=15,
                   color=colors_map.get(vibe, "gray"))
    ax.set_title(f"K-Means Clusters (k={len(np.unique(cluster_labels))}) — PCA 2D Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Vibe", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ===========================================================================
# Section 3 — Supervised Classification
# ===========================================================================

def split_data(
    X,
    y: np.ndarray,
    test_size: float = 0.2,
) -> tuple:
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Fit and save the baseline Logistic Regression model."""
    lr = LogisticRegression(
        max_iter=1000, C=1.0,
        solver="lbfgs", multi_class="multinomial",
        random_state=42,
    )
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    print("Saved: logistic_regression.joblib")
    return lr


def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """Grid-search XGBoost and save the best estimator."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [3, 5],
    }
    grid = GridSearchCV(
        xgb.XGBClassifier(eval_metric="mlogloss", random_state=42,
                           tree_method="hist", verbosity=0),
        param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=2,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV F1-macro: {grid.best_score_:.4f}")
    joblib.dump(best, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    print("Saved: xgboost_model.joblib")
    return best


def compute_learning_curve(
    X_train,
    y_train,
    save_path: str = "learning_curve.png",
) -> None:
    """Plot and save the Logistic Regression learning curve (bias-variance tradeoff)."""
    train_sizes, train_scores, val_scores = learning_curve(
        LogisticRegression(max_iter=500, random_state=42),
        X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=3, scoring="f1_macro", n_jobs=-1,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean,   val_std   = val_scores.mean(axis=1),   val_scores.std(axis=1)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="blue",   label="Train F1-macro")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue")
    plt.plot(train_sizes, val_mean,   "s-", color="orange", label="Validation F1-macro")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="orange")
    plt.title("Learning Curve — Logistic Regression", fontsize=13)
    plt.xlabel("Training Set Size")
    plt.ylabel("F1-macro Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ===========================================================================
# Section 4 — Evaluation & Diagnostics
# ===========================================================================

def evaluate_models(
    lr_model: LogisticRegression,
    xgb_model: xgb.XGBClassifier,
    X_test,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> pd.DataFrame:
    """
    Print classification reports, confusion matrices, ROC curves, and a
    summary comparison table. Saves all plots to disk.

    Returns
    -------
    pd.DataFrame with Accuracy, F1 Macro, ROC-AUC for both models.
    """
    y_pred_lr  = lr_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    print("=" * 50)
    print("LOGISTIC REGRESSION — Baseline")
    print("=" * 50)
    print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

    print("=" * 50)
    print("XGBOOST — Tuned")
    print("=" * 50)
    print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, preds, title in zip(
        axes,
        [y_pred_lr, y_pred_xgb],
        ["Logistic Regression (Baseline)", "XGBoost (Tuned)"],
    ):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("Confusion Matrices — Who Gets Confused for Whom?", fontsize=13)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=100)
    plt.close(fig)
    print("Saved: confusion_matrices.png")

    # ROC curves
    y_test_bin  = label_binarize(y_test, classes=list(range(len(le.classes_))))
    y_score_lr  = lr_model.predict_proba(X_test)
    y_score_xgb = xgb_model.predict_proba(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.tab10.colors
    for ax, scores, title in zip(
        axes,
        [y_score_lr, y_score_xgb],
        ["Logistic Regression", "XGBoost"],
    ):
        for i, class_name in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], scores[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f"{class_name} (AUC={roc_auc_val:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
        ax.set_title(f"ROC Curves — {title}", fontsize=12)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=100)
    plt.close(fig)
    print("Saved: roc_curves.png")

    # Summary table
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "XGBoost"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_lr),
            accuracy_score(y_test, y_pred_xgb),
        ],
        "F1 Macro": [
            f1_score(y_test, y_pred_lr,  average="macro"),
            f1_score(y_test, y_pred_xgb, average="macro"),
        ],
        "ROC-AUC (macro ovr)": [
            roc_auc_score(y_test_bin, y_score_lr,  multi_class="ovr", average="macro"),
            roc_auc_score(y_test_bin, y_score_xgb, multi_class="ovr", average="macro"),
        ],
    }).set_index("Model").round(4)

    print("\nModel Comparison:")
    print(comparison.to_string())
    return comparison


# ===========================================================================
# Section 5 — Inference API (used by any UI layer)
# ===========================================================================

class Artifacts:
    """Container for all trained artifacts, returned by load_artifacts()."""

    def __init__(
        self,
        tfidf:  TfidfVectorizer,
        le:     LabelEncoder,
        lr:     LogisticRegression,
        xgb_model: xgb.XGBClassifier,
        km:     KMeans,
        pca:    PCA | None,
        vibe_labels: dict[int, str],
    ):
        self.tfidf       = tfidf
        self.le          = le
        self.lr          = lr
        self.xgb_model   = xgb_model
        self.km          = km
        self.pca         = pca
        self.vibe_labels = vibe_labels
        self.classes_    = list(le.classes_)


def load_artifacts(models_dir: str = MODELS_DIR) -> Artifacts:
    """
    Load all saved model artifacts from disk.

    Raises FileNotFoundError if any required artifact is missing
    (run backend.py as a script first to train and save them).
    """
    def _load(name: str):
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run `python backend.py` first to train and save all models."
            )
        return joblib.load(path)

    pca = None
    try:
        pca = joblib.load(os.path.join(models_dir, "pca_50.joblib"))
    except FileNotFoundError:
        pass

    return Artifacts(
        tfidf      = _load("tfidf_vectorizer.joblib"),
        le         = _load("label_encoder.joblib"),
        lr         = _load("logistic_regression.joblib"),
        xgb_model  = _load("xgboost_model.joblib"),
        km         = _load("kmeans_model.joblib"),
        pca        = pca,
        vibe_labels= VIBE_LABELS,
    )


def predict(text: str, artifacts: Artifacts) -> dict:
    """
    Classify a single quote and return a structured result dict.

    Parameters
    ----------
    text      : raw dialogue string (not preprocessed)
    artifacts : loaded Artifacts from load_artifacts()

    Returns
    -------
    dict with keys:
        clean_text      str   — preprocessed version of input
        lr_probs        dict  — {character: probability} for Logistic Regression
        xgb_probs       dict  — {character: probability} for XGBoost
        lr_prediction   str   — top character (LR)
        xgb_prediction  str   — top character (XGBoost)
        cluster_id      int | None
        vibe            str | None
    """
    clean = preprocess(text)
    if not clean:
        return {
            "clean_text":     clean,
            "lr_probs":       {},
            "xgb_probs":      {},
            "lr_prediction":  None,
            "xgb_prediction": None,
            "cluster_id":     None,
            "vibe":           None,
            "error":          "Input reduced to empty after preprocessing.",
        }

    vec = artifacts.tfidf.transform([clean])

    lr_p   = artifacts.lr.predict_proba(vec)[0]
    xgb_p  = artifacts.xgb_model.predict_proba(vec)[0]
    chars  = artifacts.classes_

    cluster_id, vibe = None, None
    if artifacts.pca is not None:
        try:
            vec_dense  = vec.toarray().astype(np.float64)
            vec_pca    = artifacts.pca.transform(vec_dense)
            if np.isfinite(vec_pca).all():
                cluster_id = int(artifacts.km.predict(vec_pca)[0])
                vibe       = artifacts.vibe_labels.get(cluster_id, f"Cluster {cluster_id}")
        except Exception:
            pass

    return {
        "clean_text":     clean,
        "lr_probs":       dict(zip(chars, lr_p.tolist())),
        "xgb_probs":      dict(zip(chars, xgb_p.tolist())),
        "lr_prediction":  chars[int(np.argmax(lr_p))],
        "xgb_prediction": chars[int(np.argmax(xgb_p))],
        "cluster_id":     cluster_id,
        "vibe":           vibe,
    }


# ===========================================================================
# Main — run the full pipeline end-to-end
# ===========================================================================

def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load & EDA
    df          = load_data()
    line_counts = eda_class_imbalance(df)
    df_filtered = filter_top_characters(df, TOP_N_CHARACTERS)

    # 2. TF-IDF vectorization
    tfidf, X, le, y = build_tfidf(df_filtered, TFIDF_MAX_FEATS)
    print_top_tfidf_terms(tfidf, X, le, y)

    X_dense = X.toarray()

    # 3. Clustering
    pca_model, X_pca = reduce_pca(X_dense, PCA_COMPONENTS)
    elbow_and_silhouette(X_pca)
    km_final, cluster_labels = fit_kmeans(X_pca, X_dense, tfidf, N_CLUSTERS)

    df_filtered["cluster"] = cluster_labels
    df_filtered["vibe"]    = df_filtered["cluster"].map(VIBE_LABELS)
    print("\nCluster distribution:")
    print(df_filtered["vibe"].value_counts().to_string())

    save_cluster_plot(X_pca, cluster_labels)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Supervised models
    lr_model  = train_logistic_regression(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    compute_learning_curve(X_train, y_train)

    # 6. Evaluate
    evaluate_models(lr_model, xgb_model, X_test, y_test, le)

    print("\nAll artifacts saved to models/. Ready for UI integration.")
    print("Example usage:")
    print("  from backend import load_artifacts, predict")
    print("  arts = load_artifacts()")
    print("  result = predict(\"Bears. Beets. Battlestar Galactica.\", arts)")
    print("  print(result)")


if __name__ == "__main__":
    main()
