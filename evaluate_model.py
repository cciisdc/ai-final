"""
evaluate_model.py
=================
Evaluates the two supervised VSE neural networks from Heilmeier et al. (2020)
using pre-processed feature CSV files produced by preprocess_vse.py.

    tc_features.csv  →  evaluates the TC (Tire-Change) NN
    cc_features.csv  →  evaluates the CC (Compound-Choice) NN

Metrics match the paper:
    TC → F1 score        (paper ≈ 0.59, hybrid NN)
    CC → Accuracy        (paper ≈ 0.77)

Usage
-----
    python evaluate_model.py vse_features

If no path is given the script looks for the 'vse_features' folder in the
same directory (produced by preprocess_vse.py).
"""

import sys, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_here     = os.path.dirname(os.path.abspath(__file__))
_repo     = _here
_cand     = os.path.join(_here, "race-simulation-master")
if os.path.isdir(_cand):
    _repo = _cand

VSE_DIR  = os.path.join(_repo, "racesim", "input", "vse")
TC_MODEL = os.path.join(VSE_DIR, "nn_supervised_tirechange.tflite")
CC_MODEL = os.path.join(VSE_DIR, "nn_supervised_compoundchoice.tflite")
TC_PREP  = os.path.join(VSE_DIR, "preprocessor_supervised_tirechange.pkl")
CC_PREP  = os.path.join(VSE_DIR, "preprocessor_supervised_compoundchoice.pkl")

FEAT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, "vse_features")
TC_CSV   = os.path.join(FEAT_DIR, "tc_features.csv")
CC_CSV   = os.path.join(FEAT_DIR, "cc_features.csv")

OUT_DIR  = os.path.join(_here, "evaluation_results")


# ---------------------------------------------------------------------------
# Model / preprocessor loaders
# ---------------------------------------------------------------------------

def load_preprocessor(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_tflite(path: str) -> dict:
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return {
        "interpreter":  interp,
        "input_index":  interp.get_input_details()[0]["index"],
        "output_index": interp.get_output_details()[0]["index"],
        "input_shape":  interp.get_input_details()[0]["shape"],
    }


# ---------------------------------------------------------------------------
# TC inference  (rolling LSTM window, exactly as in vse_supervised.py)
# ---------------------------------------------------------------------------

def predict_tc(model: dict, preprocessor, tc_df: pd.DataFrame):
    """
    Run tire-change inference lap-by-lap per driver, maintaining the rolling
    time-series window exactly as the simulation does.
    Returns (y_true, y_pred, probs).
    """
    no_ts = model["input_shape"][1]   # number of time steps from model

    feature_cols = [
        "tire_age_progress", "race_progress", "pos_encoded",
        "rel_compound_num", "fcy_status", "remaining_pits",
        "tirechange_pursuer", "location_cat", "close_ahead",
    ]

    tc_df = tc_df.sort_values(["race_id", "driver_id", "lapno"]).copy()
    y_true_all, y_pred_all, probs_all = [], [], []

    for (race_id, driver_id), grp in tc_df.groupby(["race_id", "driver_id"]):
        grp    = grp.sort_values("lapno").reset_index(drop=True)
        X_raw  = grp[feature_cols].values.astype(np.float32)

        # Initialise rolling window (paper: first row repeated, FCY zeroed)
        init_row      = X_raw[0:1].copy()
        init_row[0, 4] = 0.0   # zero out FCY for prior steps
        init_conv     = preprocessor.transform(init_row, dtype_out=np.float32)[0]
        first_conv    = preprocessor.transform(X_raw[0:1], dtype_out=np.float32)[0]
        window        = np.tile(init_conv, (no_ts, 1))
        window[-1]    = first_conv

        for i in range(len(grp)):
            row_conv    = preprocessor.transform(X_raw[i:i+1], dtype_out=np.float32)[0]
            window      = np.roll(window, -1, axis=0)
            window[-1]  = row_conv

            inp = window[np.newaxis].astype(np.float32)
            model["interpreter"].set_tensor(model["input_index"], inp)
            model["interpreter"].invoke()
            prob = float(np.ravel(
                model["interpreter"].get_tensor(model["output_index"])
            )[0])

            y_true_all.append(int(grp.at[i, "is_pit"]))
            y_pred_all.append(1 if prob >= 0.5 else 0)
            probs_all.append(prob)

    return (
        np.array(y_true_all),
        np.array(y_pred_all),
        np.array(probs_all),
    )


# ---------------------------------------------------------------------------
# CC inference
# ---------------------------------------------------------------------------

def predict_cc(model: dict, preprocessor, cc_df: pd.DataFrame):
    """Run compound-choice inference for all pit-stop rows."""
    feature_cols = [
        "race_progress", "location_num", "rel_compound_num",
        "remaining_pits", "used_2_compounds", "n_dry",
    ]

    cc_df = cc_df.copy()
    loc_map = preprocessor.cat_dict["location"]
    cc_df["location_num"] = cc_df["location"].map(loc_map).fillna(1).astype(int)

    X_raw  = cc_df[feature_cols].values.astype(np.float32)
    y_true = cc_df["cc_label"].values.astype(int)
    y_pred = []

    for i in range(len(X_raw)):
        row_conv = preprocessor.transform(X_raw[i:i+1], dtype_out=np.float32)
        model["interpreter"].set_tensor(model["input_index"], row_conv)
        model["interpreter"].invoke()
        out = model["interpreter"].get_tensor(model["output_index"])[0]
        y_pred.append(int(np.argmax(out)))

    return y_true, np.array(y_pred)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = {
    "bg":      "#0d0d0f", "panel":   "#14151a", "border":  "#2a2d3a",
    "accent1": "#e8c547", "accent2": "#4fc3f7", "accent3": "#ef5350",
    "text":    "#e8e8ec", "subtext": "#8890a4",
}

def _dark(fig, axes):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
        for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            lbl.set_color(PALETTE["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["border"])


def plot_confusion_matrix(cm, labels, title, outpath):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _dark(fig, ax)
    cmap = sns.light_palette(PALETTE["accent1"], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor=PALETTE["border"], ax=ax,
                cbar_kws={"shrink": 0.75})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Truth", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def plot_metrics_bar(metrics, title, outpath, paper_ref=None):
    labels, values = list(metrics.keys()), list(metrics.values())
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _dark(fig, ax)
    colors = [PALETTE["accent1"], PALETTE["accent2"],
               PALETTE["accent3"], "#a29bfe", "#55efc4"]
    bars = ax.barh(labels, values, color=colors[:len(labels)],
                   edgecolor=PALETTE["border"], linewidth=0.6, height=0.55)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                color=PALETTE["text"], fontsize=10)
    if paper_ref:
        ax.axvline(paper_ref, color=PALETTE["accent3"], linewidth=1.4,
                   linestyle="--", label=f"Paper target ≈ {paper_ref}")
        ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text"], fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def plot_prob_hist(probs, y_true, title, outpath):
    fig, ax = plt.subplots(figsize=(7, 4))
    _dark(fig, ax)
    bins = np.linspace(0, 1, 30)
    ax.hist(probs[y_true == 0], bins=bins, alpha=0.65,
            color=PALETTE["accent2"], label="No pit stop")
    ax.hist(probs[y_true == 1], bins=bins, alpha=0.75,
            color=PALETTE["accent1"], label="Pit stop")
    ax.axvline(0.5, color=PALETTE["accent3"], linestyle="--",
               linewidth=1.4, label="Threshold 0.50")
    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_tc(tc_df: pd.DataFrame, out_dir: str) -> dict:
    print("\n" + "=" * 60)
    print("  TIRE-CHANGE (TC) NN — Evaluation")
    print("=" * 60)

    preprocessor = load_preprocessor(TC_PREP)
    model        = load_tflite(TC_MODEL)

    print(f"  Running inference on {len(tc_df):,} laps …")
    y_true, y_pred, probs = predict_tc(model, preprocessor, tc_df)

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0

    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * prec * rec / (prec + rec)
        if (prec + rec) > 0
        else 0
    )

    print(f"\n  Results on {len(y_true):,} samples:")
    print(f"    Accuracy   : {acc:.4f}")
    print(f"    Precision  : {prec:.4f}")
    print(f"    Recall     : {rec:.4f}")
    print(f"    F1 Score   : {f1:.4f}")
    print(f"\n  Confusion Matrix (rows=Truth, cols=Predicted):")
    print(f"                   Pred: No Pit   Pred: Pit")
    print(f"    Truth: No Pit  {cm[0,0]:10d}  {cm[0,1]:10d}")
    print(f"    Truth: Pit     {cm[1,0]:10d}  {cm[1,1]:10d}")

    plot_confusion_matrix(
        cm, ["No pit stop", "Pit stop"],
        "TC — Confusion Matrix",
        os.path.join(out_dir, "tc_confusion_matrix.png")
    )
    plot_metrics_bar(
        {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1},
        "TC — Metric Summary",
        os.path.join(out_dir, "tc_metrics_bar.png"),
        paper_ref=0.59
    )
    plot_prob_hist(
        probs, y_true,
        "TC — Predicted Probability Distribution",
        os.path.join(out_dir, "tc_prob_histogram.png")
    )

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}


def evaluate_cc(cc_df: pd.DataFrame, out_dir: str) -> dict:
    print("\n" + "=" * 60)
    print("  COMPOUND-CHOICE (CC) NN — Evaluation")
    print("=" * 60)

    preprocessor = load_preprocessor(CC_PREP)
    model        = load_tflite(CC_MODEL)

    print(f"  Running inference on {len(cc_df):,} pit-stop rows …")
    y_true, y_pred = predict_cc(model, preprocessor, cc_df)

    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    rep = classification_report(
        y_true, y_pred,
        target_names=["Hard", "Medium", "Soft"],
        zero_division=0
    )

    print(f"\n  Results on {len(y_true):,} samples:")
    print(f"    Accuracy : {acc:.4f}")
    print(f"\n  Per-class report:\n{rep}")
    print("  Confusion Matrix (rows=Truth, cols=Predicted):")
    print(pd.DataFrame(
        cm,
        index=["True Hard", "True Med", "True Soft"],
        columns=["Pred Hard", "Pred Med", "Pred Soft"]
    ).to_string())

    per_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    plot_confusion_matrix(
        cm, ["Hard", "Medium", "Soft"],
        "CC — Confusion Matrix",
        os.path.join(out_dir, "cc_confusion_matrix.png")
    )
    plot_metrics_bar(
        {"Accuracy": acc, "F1 Hard": per_f1[0],
         "F1 Medium": per_f1[1], "F1 Soft": per_f1[2]},
        "CC — Metric Summary",
        os.path.join(out_dir, "cc_metrics_bar.png"),
        paper_ref=0.77
    )

    return {"accuracy": acc, "cm": cm, "per_class_f1": per_f1}




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for path, name in [(TC_CSV, "tc_features.csv"), (CC_CSV, "cc_features.csv")]:
        if not os.path.exists(path):
            print(f"\nERROR: '{name}' not found at '{path}'")
            print("Run preprocess_vse.py first to generate the feature CSV files.")
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nLoading feature files from: {FEAT_DIR}")
    print(f"Outputs will be saved to  : {OUT_DIR}")

    tc_df = pd.read_csv(TC_CSV)
    cc_df = pd.read_csv(CC_CSV)

    # --- hold out 2019 as test set ---
    tc_df = tc_df[tc_df["season"] == 2019].reset_index(drop=True)
    cc_df = cc_df[cc_df["season"] == 2019].reset_index(drop=True)

    print(f"  TC rows loaded: {len(tc_df):,}")
    print(f"  CC rows loaded: {len(cc_df):,}")

    tc_res = evaluate_tc(tc_df, OUT_DIR)
    cc_res = evaluate_cc(cc_df, OUT_DIR)

    print(f"\nDone. All outputs saved to '{OUT_DIR}'")