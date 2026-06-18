"""
app.py  —  VSE Web Interface
==============================
Flask web app for VSE
Provides two features:
  1. Evaluate the TC/CC neural networks on a chosen season
  2. Run the Monte Carlo race simulation for a chosen race .ini
"""

import os, sys, json, base64, subprocess, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE    = os.path.dirname(os.path.abspath(__file__))
VSE_DIR  = os.path.join(_BASE, "racesim", "input", "vse")
PAR_DIR  = os.path.join(_BASE, "racesim", "input", "parameters")
FEAT_DIR = os.path.join(_BASE, "vse_features")

TC_MODEL = os.path.join(VSE_DIR, "nn_supervised_tirechange.tflite")
CC_MODEL = os.path.join(VSE_DIR, "nn_supervised_compoundchoice.tflite")
TC_PREP  = os.path.join(VSE_DIR, "preprocessor_supervised_tirechange.pkl")
CC_PREP  = os.path.join(VSE_DIR, "preprocessor_supervised_compoundchoice.pkl")
TC_CSV   = os.path.join(FEAT_DIR, "tc_features.csv")
CC_CSV   = os.path.join(FEAT_DIR, "cc_features.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


DARK = {
    "bg": "#0d0d0f", "panel": "#14151a", "border": "#2a2d3a",
    "gold": "#e8c547", "blue": "#4fc3f7", "red": "#ef5350",
    "text": "#e8e8ec", "sub": "#8890a4",
}

# ---------------------------------------------------------------------------
# Season-correct relative compound (A1-A7) -> display name mapping.
# ---------------------------------------------------------------------------
SEASON_COMPOUND_NAMES = {
    2014: {"A1": "Hard", "A2": "Medium", "A3": "Soft", "A4": "Supersoft"},
    2015: {"A1": "Hard", "A2": "Medium", "A3": "Soft", "A4": "Supersoft"},
    2016: {"A1": "Hard", "A2": "Medium", "A3": "Soft", "A4": "Supersoft", "A5": "Ultrasoft"},
    2017: {"A1": "Hard", "A2": "Medium", "A3": "Soft", "A4": "Supersoft", "A5": "Ultrasoft"},
    2018: {"A1": "Superhard", "A2": "Hard", "A3": "Medium", "A4": "Soft",
           "A5": "Supersoft", "A6": "Ultrasoft", "A7": "Hypersoft"},
    2019: {"A2": "C1", "A3": "C2", "A4": "C3", "A6": "C4", "A7": "C5"},
}
# non-dry compounds are season-independent
_STATIC_COMPOUND_NAMES = {"I": "Intermediate", "W": "Wet"}


def get_season_from_race_name(race: str) -> int:
    """Extract the trailing 4-digit year from a race identifier like
    'Austin_2014' -> 2014. Falls back to 2019 (most permissive table) if
    no year can be parsed."""
    import re
    m = re.search(r'(20\d{2})', race)
    return int(m.group(1)) if m else 2019


def code_to_compound_name(code: str, season: int) -> str:
    """Converts a raw relative compound code (e.g. 'A2') to its
    season-correct display name (e.g. 'Medium' for 2014, 'Ultrasoft' for
    2016). Falls back to the raw code itself if unmapped."""
    table = {**_STATIC_COMPOUND_NAMES, **SEASON_COMPOUND_NAMES.get(season, {})}
    return table.get(code, code)


def build_strategy_string(raw_strategy: str, season: int) -> str:
    """Parses the raw strategy_info cell as printed by
    _race_raceanalysis.py's original print_result() - a stringified list
    of [lap, code] pairs like "[0, 'A3'], [28, 'A2']" - and converts it
    into a season-correct, human-readable strategy timeline string like
    '[lap 0: Soft] -> [lap 28: Medium]'."""
    import re
    pairs = re.findall(r"\[(\d+),\s*'([^']+)'\]", raw_strategy)
    if not pairs:
        return raw_strategy  # fallback: unrecognized format, show as-is
    segments = [
        f"[lap {lap}: {code_to_compound_name(code, season)}]"
        for lap, code in pairs
    ]
    return " -> ".join(segments)

def build_cc_probs(race_season, available_compounds, driver_map, rows_raw_strategy):
    
    cc_log_path = os.path.join(_BASE, "vse_cc_probs.csv")
    if not os.path.exists(cc_log_path) or not driver_map:
        return None

    try:
        sorted_compounds = sorted([c for c in available_compounds if c not in ("I", "W")])
        neuron_labels = []
        for i in range(3):
            if i < len(sorted_compounds):
                neuron_labels.append(code_to_compound_name(sorted_compounds[i], race_season))
            else:
                neuron_labels.append(None)

        cc_df = pd.read_csv(cc_log_path)
        cc_df["driver"] = cc_df["driver_idx"].astype(str).map(driver_map)

        cc_probs = {}
        for driver, grp in cc_df.groupby("driver"):
            # actual pit stops for this driver, in race order, skipping
            # the lap-0 starting compound (not a pit stop decision)
            pit_pairs = [(lap, code) for lap, code in rows_raw_strategy.get(driver, []) if lap != 0]

            entries = []
            grp_sorted = grp.sort_values("lap").reset_index(drop=True)
            for i, r in grp_sorted.iterrows():
                probs = [float(r["prob_relA"]), float(r["prob_relB"]), float(r["prob_relC"])]
                options = [
                    {"name": neuron_labels[j], "prob": round(probs[j], 4)}
                    for j in range(3) if neuron_labels[j] is not None
                ]
                # match this CC decision to the Nth actual pit stop by order
                if i < len(pit_pairs):
                    actual_lap, chosen_code = pit_pairs[i]
                    chosen_name = code_to_compound_name(chosen_code, race_season)
                else:
                    actual_lap, chosen_name = None, None
                entries.append({"lap": actual_lap, "options": options, "chosen": chosen_name})
            cc_probs[driver] = entries

        return cc_probs
    except Exception:
        return None

TEAM_COLORS = {
    "HAM": "#00D2BE", "BOT": "#00D2BE", "ROS": "#00D2BE",            # Mercedes
    "VET": "#E10600", "RAI": "#E10600", "LEC": "#E10600",            # Ferrari
    "VER": "#1E41FF", "RIC": "#1E41FF", "DAN": "#1E41FF",
    "GAS": "#1E41FF", "ALB": "#1E41FF",                              # Red Bull
    "NOR": "#FF8700", "SAI": "#FF8700", "HUL": "#FF8700",
    "MAG": "#FF8700", "VAN": "#FF8700", "ALO": "#FF8700",            # McLaren/Renault era blend
    "PER": "#2293D1", "OCO": "#2293D1", "STR": "#2293D1",
    "MAS": "#2293D1",                                                 # Force India/Racing Point
    "KVY": "#1660AD", "GRO": "#1660AD",                               # Toro Rosso/Haas blend
    "KUB": "#9B0000",                                                 # Williams
    "RUS": "#9B0000",
    "GIO": "#9C9FA2", "ERI": "#9C9FA2",                               # Sauber/Alfa Romeo
    "WEH": "#9C9FA2",
}

TEAM_NAMES = {
    "HAM": "Mercedes", "BOT": "Mercedes", "ROS": "Mercedes",
    "VET": "Ferrari", "RAI": "Ferrari", "LEC": "Ferrari",
    "VER": "Red Bull Racing", "RIC": "Red Bull Racing", "DAN": "Red Bull Racing",
    "GAS": "Red Bull Racing", "ALB": "Red Bull Racing",
    "NOR": "McLaren", "SAI": "McLaren / Renault", "HUL": "Renault",
    "MAG": "Haas", "VAN": "McLaren", "ALO": "McLaren",
    "PER": "Force India / Racing Point", "OCO": "Force India / Racing Point",
    "STR": "Williams / Racing Point", "MAS": "Williams",
    "KVY": "Toro Rosso", "GRO": "Haas",
    "KUB": "Williams", "RUS": "Williams",
    "GIO": "Sauber / Alfa Romeo", "ERI": "Sauber", "WEH": "Sauber",
}

def _style(fig, axes):
    fig.patch.set_facecolor(DARK["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(DARK["panel"])
        ax.tick_params(colors=DARK["sub"], labelsize=9)
        for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            lbl.set_color(DARK["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(DARK["border"])


def load_preprocessor(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return {
        "interpreter":  interp,
        "input_index":  interp.get_input_details()[0]["index"],
        "output_index": interp.get_output_details()[0]["index"],
        "input_shape":  interp.get_input_details()[0]["shape"],
    }

# ---------------------------------------------------------------------------
# TC inference
# ---------------------------------------------------------------------------

def predict_tc(model, preprocessor, tc_df):
    no_ts = model["input_shape"][1]
    feature_cols = [
        "tire_age_progress", "race_progress", "pos_encoded",
        "rel_compound_num", "fcy_status", "remaining_pits",
        "tirechange_pursuer", "location_cat", "close_ahead",
    ]
    tc_df = tc_df.sort_values(["race_id", "driver_id", "lapno"]).copy()
    y_true_all, y_pred_all, probs_all = [], [], []

    for (_, _), grp in tc_df.groupby(["race_id", "driver_id"]):
        grp   = grp.sort_values("lapno").reset_index(drop=True)
        X_raw = grp[feature_cols].values.astype(np.float32)
        init_row       = X_raw[0:1].copy(); init_row[0, 4] = 0.0
        init_conv      = preprocessor.transform(init_row, dtype_out=np.float32)[0]
        first_conv     = preprocessor.transform(X_raw[0:1], dtype_out=np.float32)[0]
        window         = np.tile(init_conv, (no_ts, 1)); window[-1] = first_conv

        for i in range(len(grp)):
            row_conv = preprocessor.transform(X_raw[i:i+1], dtype_out=np.float32)[0]
            window   = np.roll(window, -1, axis=0); window[-1] = row_conv
            inp      = window[np.newaxis].astype(np.float32)
            model["interpreter"].set_tensor(model["input_index"], inp)
            model["interpreter"].invoke()
            prob = float(np.ravel(model["interpreter"].get_tensor(model["output_index"]))[0])
            y_true_all.append(int(grp.at[i, "is_pit"]))
            y_pred_all.append(1 if prob >= 0.5 else 0)
            probs_all.append(prob)

    return np.array(y_true_all), np.array(y_pred_all), np.array(probs_all)


def predict_cc(model, preprocessor, cc_df):
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
# Chart generators
# ---------------------------------------------------------------------------

def make_cm_chart(cm, labels, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    _style(fig, ax)
    cmap = sns.light_palette(DARK["gold"], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor=DARK["border"], ax=ax,
                cbar_kws={"shrink": 0.75})
    ax.set_xlabel("Predicted", fontsize=10, color=DARK["text"])
    ax.set_ylabel("Actual", fontsize=10, color=DARK["text"])
    ax.set_title(title, fontsize=12, pad=10, color=DARK["text"])
    plt.tight_layout()
    return _fig_to_b64(fig)


def make_metrics_chart(metrics, title, paper_ref=None):
    labels, values = list(metrics.keys()), list(metrics.values())
    colors = [DARK["gold"], DARK["blue"], DARK["red"], "#a29bfe", "#55efc4"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    _style(fig, ax)
    bars = ax.barh(labels, values, color=colors[:len(labels)],
                   edgecolor=DARK["border"], linewidth=0.5, height=0.5)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                color=DARK["text"], fontsize=9)
    if paper_ref:
        ax.axvline(paper_ref, color=DARK["red"], linewidth=1.3,
                   linestyle="--", label=f"Paper ref ≈ {paper_ref}")
        ax.legend(facecolor=DARK["panel"], edgecolor=DARK["border"],
                  labelcolor=DARK["text"], fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title(title, fontsize=12, pad=8)
    plt.tight_layout()
    return _fig_to_b64(fig)

# ---------------------------------------------------------------------------
# Parse simulation output
# ---------------------------------------------------------------------------

def parse_sim_output(raw: str):
    lines = raw.splitlines()
    rows, fcy, retirements = [], [], []
    in_table = False

    for line in lines:
        if line.startswith("RESULT: Simulation result:"):
            in_table = True; continue
        if in_table:
            stripped = line.strip()
            if not stripped or stripped.startswith("pos") or stripped.startswith("---"):
                continue
            if stripped.startswith("RESULT:") or stripped.startswith("INFO:"):
                in_table = False
            else:
                # the first 9 whitespace-separated tokens are the fixed
                # columns; everything after that (rejoined) is the raw
                # strategy_info string, which may itself contain spaces
                parts = stripped.split(None, 9)
                if len(parts) >= 9:
                    driver = parts[0]
                    pos    = parts[1]
                    carno  = parts[2]
                    t_race = parts[3]
                    gap    = parts[4]
                    interval = parts[5]
                    best   = parts[6]
                    laps   = parts[7]
                    status = parts[8]
                    raw_strategy = parts[9] if len(parts) > 9 else ""
                    rows.append({
                        "driver": driver, "pos": pos, "car": carno,
                        "t_race": t_race, "gap": gap, "interval": interval,
                        "best_lap": best, "laps": laps,
                        "status": status, "raw_strategy": raw_strategy
                    })
        if "RESULT: No FCY phases" in line:
            fcy = []
        if "RESULT: FCY phases:" in line:
            fcy = line.split(":", 1)[1].strip()
        if "RESULT: Retirements:" in line:
            retirements = line.split(":", 1)[1].strip()

    return rows, fcy, retirements

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

BROKEN_RACES = {
    "Austin_2015", "Baku_2016", "Baku_2017", "Budapest_2014",
    "Budapest_2015", "Budapest_2016", "Hockenheim_2019", "KualaLumpur_2014",
    "Melbourne_2018", "Melbourne_2019", "MexicoCity_2016", "MonteCarlo_2015",
    "MonteCarlo_2016", "Montreal_2015", "Monza_2014", "Monza_2015",
    "Monza_2016", "Monza_2019", "SaoPaulo_2016", "SaoPaulo_2018",
    "Shanghai_2014", "Shanghai_2017", "Silverstone_2016", "Silverstone_2018",
    "Singapore_2015", "Singapore_2017", "Sochi_2014", "Sochi_2016",
    "Sochi_2019", "Spa_2014", "Spielberg_2017", "Suzuka_2014",
    "Suzuka_2015", "Suzuka_2016"
}


@app.route("/")
def index():
    ini_files = sorted([
        f.replace("pars_", "").replace(".ini", "")
        for f in os.listdir(PAR_DIR)
        if f.endswith(".ini") and f != "pars_mcs.ini" and
        f.replace("pars_", "").replace(".ini", "") not in BROKEN_RACES
    ])
    return render_template("index.html", ini_files=ini_files)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    season = 2019

    tc_df = pd.read_csv(TC_CSV)
    cc_df = pd.read_csv(CC_CSV)
    tc_df = tc_df[tc_df["season"] == season].reset_index(drop=True)
    cc_df = cc_df[cc_df["season"] == season].reset_index(drop=True)

    if len(tc_df) == 0:
        return jsonify({"error": f"No data found for season {season}"}), 400

    # --- TC ---
    tc_prep  = load_preprocessor(TC_PREP)
    tc_model = load_tflite(TC_MODEL)
    y_true_tc, y_pred_tc, probs_tc = predict_tc(tc_model, tc_prep, tc_df)

    f1   = float(f1_score(y_true_tc, y_pred_tc, zero_division=0))
    prec = float(precision_score(y_true_tc, y_pred_tc, zero_division=0))
    rec  = float(recall_score(y_true_tc, y_pred_tc, zero_division=0))
    acc  = float(accuracy_score(y_true_tc, y_pred_tc))
    cm_tc = confusion_matrix(y_true_tc, y_pred_tc)

    tn, fp, fn, tp = cm_tc.ravel()

    tc_cm_img  = make_cm_chart(cm_tc, ["No Pit", "Pit Stop"], "TC — Confusion Matrix")
    tc_bar_img = make_metrics_chart(
        {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1},
        "TC — Metric Summary", paper_ref=0.59
    )

    # --- CC ---
    cc_prep  = load_preprocessor(CC_PREP)
    cc_model = load_tflite(CC_MODEL)
    y_true_cc, y_pred_cc = predict_cc(cc_model, cc_prep, cc_df)

    cc_acc   = float(accuracy_score(y_true_cc, y_pred_cc))
    cm_cc    = confusion_matrix(y_true_cc, y_pred_cc, labels=[0, 1, 2])
    per_f1   = f1_score(y_true_cc, y_pred_cc, average=None, labels=[0, 1, 2], zero_division=0)

    cc_cm_img  = make_cm_chart(cm_cc, ["Hard", "Medium", "Soft"], "CC — Confusion Matrix")
    cc_bar_img = make_metrics_chart(
        {"Accuracy": cc_acc, "F1 Hard": float(per_f1[0]),
         "F1 Medium": float(per_f1[1]), "F1 Soft": float(per_f1[2])},
        "CC — Metric Summary", paper_ref=0.77
    )

    return jsonify({
        "season": season,
        "tc": {
            "samples": int(len(y_true_tc)),
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "cm_img": tc_cm_img, "bar_img": tc_bar_img,
        },
        "cc": {
            "samples": int(len(y_true_cc)),
            "accuracy": round(cc_acc, 4),
            "f1_hard": round(float(per_f1[0]), 4),
            "f1_medium": round(float(per_f1[1]), 4),
            "f1_soft": round(float(per_f1[2]), 4),
            "cm_img": cc_cm_img, "bar_img": cc_bar_img,
        }
    })


@app.route("/simulate", methods=["POST"])
def simulate():
    race = request.json.get("race")
    if not race:
        return jsonify({"error": "No race selected"}), 400

    ini_path = os.path.join(PAR_DIR, f"pars_{race}.ini")
    if not os.path.exists(ini_path):
        return jsonify({"error": f"Parameter file not found: pars_{race}.ini"}), 404
    
    import configparser
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read(ini_path)
    try:
        vse_pars_raw = json.loads(cfg_parser.get('VSE_PARS', 'vse_pars'))
        available_compounds = vse_pars_raw.get("available_compounds", [])
    except Exception:
        available_compounds = []

    main_path = os.path.join(_BASE, "main_racesim.py")

    with open(main_path, "r") as f:
        original = f.read()

    import re
    patched = re.sub(
        r'race_pars_file_\s*=\s*["\'].*?["\']',
        f'race_pars_file_ = "pars_{race}.ini"',
        original
    )
    patched = re.sub(
        r'"use_vse"\s*:\s*False',
        '"use_vse": True',
        patched
    )

    tmp_path = os.path.join(_BASE, "_tmp_racesim.py")
    with open(tmp_path, "w") as f:
        f.write(patched)

    # clear stale VSE logs from any previous run before this one starts
    for stale in ["vse_live_probs.csv", "vse_driver_map.json"]:
        stale_path = os.path.join(_BASE, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = _BASE + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=120,
            cwd=_BASE, env=env, encoding="utf-8", errors="replace"
        )
        output = result.stdout + result.stderr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    rows, fcy, retirements = parse_sim_output(output)

    #build season-correct, human-readable strategy strings from the raw [lap, code] pairs printed by racesim's original print_result() ---
    import re as _re
    race_season = get_season_from_race_name(race)
    rows_raw_pairs = {}
    for row in rows:
        raw = row["raw_strategy"]
        rows_raw_pairs[row["driver"]] = [
            (int(lap), code) for lap, code in _re.findall(r"\[(\d+),\s*'([^']+)'\]", raw)
        ]
        row["strategy"] = build_strategy_string(row.pop("raw_strategy"), race_season)

    #attach team color per driver (2014-2019 era constructors) ---
    for row in rows:
        row["team_color"] = TEAM_COLORS.get(row["driver"], "#5a5a66")
        row["team_name"] = TEAM_NAMES.get(row["driver"], "")

    #read live VSE probability log + driver index map (if VSE was used) ---
    prob_log_path = os.path.join(_BASE, "vse_live_probs.csv")
    driver_map_path = os.path.join(_BASE, "vse_driver_map.json")
    vse_probs = None

    if os.path.exists(prob_log_path) and os.path.exists(driver_map_path):
        try:
            with open(driver_map_path, "r") as f:
                driver_map = json.load(f)  # {"0": "RIC", "1": "NOR", ...}

            prob_df = pd.read_csv(prob_log_path)
            # Normalize lap counter: make_decision can be called more than once per simulated lap (e.g. FCY rechecks) -> collapse to last call per (lap, driver)
            prob_df = prob_df.drop_duplicates(subset=["lap", "driver_idx"], keep="last")
            prob_df["driver"] = prob_df["driver_idx"].astype(str).map(driver_map)

            vse_probs = {}
            for driver, grp in prob_df.groupby("driver"):
                grp_sorted = grp.sort_values("lap")
                vse_probs[driver] = {
                    "laps": grp_sorted["lap"].tolist(),
                    "probs": grp_sorted["pitstop_prob"].round(4).tolist(),
                    "decisions": grp_sorted["decision"].tolist(),
                }
        except Exception:
            vse_probs = None
    cc_probs = build_cc_probs(race_season, available_compounds, driver_map if 'driver_map' in dir() else {}, rows_raw_pairs)
    return jsonify({
        "race": race,
        "rows": rows,
        "fcy": fcy,
        "retirements": retirements,
        "raw": output,
        "vse_probs": vse_probs,
        "cc_probs": cc_probs
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)