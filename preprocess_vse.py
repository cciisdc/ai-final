"""
Usage
-----
    python preprocess_vse.py F1_timingdata_2014_2019.sqlite

"""

import sys, os, warnings
import numpy as np
import pandas as pd
import sqlite3

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_defaults = [
    os.path.join(_here, "F1_timingdata_2014_2019.sqlite"),
    os.path.join(_here, "..", "F1_timingdata_2014_2019.sqlite"),
]
DB_PATH = sys.argv[1] if len(sys.argv) > 1 else next(
    (p for p in _defaults if os.path.exists(p)), _defaults[0]
)

# ---------------------------------------------------------------------------
# Track category mapping  (from Heilmeier et al. parameter files)
# ---------------------------------------------------------------------------
TRACK_CATEGORY = {
    "Austin": 2, "Baku": 2, "Budapest": 2, "Catalunya": 1,
    "Hockenheim": 2, "KualaLumpur": 2, "LeCastellet": 2,
    "Melbourne": 2, "MexicoCity": 2, "MonteCarlo": 3,
    "Montreal": 3, "Monza": 2, "Sakhir": 1, "SaoPaulo": 1,
    "Shanghai": 2, "Silverstone": 1, "Singapore": 3,
    "Sochi": 2, "Spa": 1, "Spielberg": 2, "Suzuka": 1, "YasMarina": 3,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_avail_dry(avail_str: str) -> list:
    """Return sorted list of dry compound names (A1=hardest … A7=softest)."""
    if not avail_str:
        return []
    parts = [c.strip() for c in str(avail_str).split(",")]
    return sorted([c for c in parts if c.startswith("A")])


def compound_to_rel(compound: str, avail_dry: list) -> int:
    """
    Convert compound string to relative label expected by the NNs.
      3-compound race: 0=hard, 1=medium, 2=soft
      2-compound race: 1=medium, 2=soft  (hard suppressed, as in paper)
    Returns -1 if compound not in available list.
    """
    try:
        idx = avail_dry.index(compound)
        return idx + 1 if len(avail_dry) == 2 else idx
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Step 1 – Load raw tables
# ---------------------------------------------------------------------------

def load_raw(db_path: str):
    print(f"\n[1/5] Loading database: {db_path}")
    conn = sqlite3.connect(db_path)
    laps_df  = pd.read_sql("SELECT * FROM laps",      conn)
    races_df = pd.read_sql("SELECT * FROM races",     conn)
    fcy_df   = pd.read_sql("SELECT * FROM fcyphases", conn)
    sf_df    = pd.read_sql(
        "SELECT race_id, driver_id, resultposition FROM starterfields", conn
    )
    conn.close()
    print(f"    Raw laps  : {len(laps_df):,}")
    print(f"    Races     : {len(races_df)}")
    return laps_df, races_df, fcy_df, sf_df


# ---------------------------------------------------------------------------
# Step 2 – Race-level and driver-level filters
# ---------------------------------------------------------------------------

def apply_filters(laps_df, races_df, sf_df):
    print("\n[2/5] Applying filters …")

    # Merge race metadata
    races_df = races_df.rename(columns={"id": "race_id"})
    laps_df = laps_df.merge(
        races_df[["race_id", "season", "location", "nolaps", "availablecompounds"]],
        on="race_id", how="left"
    )

    # Parse available dry compounds
    laps_df["avail_dry"] = laps_df["availablecompounds"].apply(parse_avail_dry)
    laps_df["n_dry"]     = laps_df["avail_dry"].apply(len)

    # Remove wet races (any lap using I or W compound)
    wet_ids = set(laps_df[laps_df["compound"].isin(["I", "W"])]["race_id"])
    print(f"    Wet races removed : {len(wet_ids)}")
    laps_df = laps_df[~laps_df["race_id"].isin(wet_ids)].copy()

    # Type fixes
    for col in ["laptime", "pitstopduration", "position", "tireage", "interval"]:
        laps_df[col] = pd.to_numeric(laps_df[col], errors="coerce")

    # Remove formation lap (lap 0)
    laps_df = laps_df[laps_df["lapno"] >= 1].copy()
    print(f"    After removing lap 0          : {len(laps_df):,} laps")

    # Remove anomalous lap times > 200 s and pit stop durations > 50 s
    laps_df = laps_df[~((laps_df["laptime"].notna()) & (laps_df["laptime"] > 200))].copy()
    laps_df = laps_df[~((laps_df["pitstopduration"].notna()) &
                         (laps_df["pitstopduration"] > 50))].copy()
    print(f"    After anomalous time filter   : {len(laps_df):,} laps")

    # Remove drivers with > 3 pit stops in a race
    pit_mask   = laps_df["pitstopduration"].notna() & (laps_df["pitstopduration"] > 0)
    pit_counts = (laps_df[pit_mask]
                  .groupby(["race_id", "driver_id"])
                  .size()
                  .reset_index(name="n_pits"))
    too_many = pit_counts[pit_counts["n_pits"] > 3][["race_id", "driver_id"]]
    too_many["_excl"] = True
    laps_df = laps_df.merge(too_many, on=["race_id", "driver_id"], how="left")
    laps_df = laps_df[laps_df["_excl"].isna()].drop(columns=["_excl"])
    print(f"    After >3 pit stop filter      : {len(laps_df):,} laps")

    # Race progress column
    laps_df["race_progress"] = laps_df["lapno"] / laps_df["nolaps"]

    # Binary pit label
    laps_df["is_pit"] = (
        laps_df["pitstopduration"].notna() & (laps_df["pitstopduration"] > 0)
    ).astype(int)

    # Remove final pit stop if race progress > 90%  (paper filter)
    pit_laps = laps_df[laps_df["is_pit"] == 1].copy()
    last_pit = (pit_laps.sort_values("lapno")
                .groupby(["race_id", "driver_id"])["lapno"].last()
                .reset_index(name="last_pit_lap"))
    laps_df = laps_df.merge(last_pit, on=["race_id", "driver_id"], how="left")
    late_mask = ((laps_df["lapno"] == laps_df["last_pit_lap"]) &
                 (laps_df["race_progress"] > 0.9))
    laps_df = laps_df[~late_mask].copy()
    laps_df["is_pit"] = (
        laps_df["pitstopduration"].notna() & (laps_df["pitstopduration"] > 0)
    ).astype(int)
    print(f"    After late pit stop filter    : {len(laps_df):,} laps")

    # Merge finishing positions
    sf_df["resultposition"] = pd.to_numeric(sf_df["resultposition"], errors="coerce")
    laps_df = laps_df.merge(
        sf_df.rename(columns={"resultposition": "result_position"}),
        on=["race_id", "driver_id"], how="left"
    )

    return laps_df


# ---------------------------------------------------------------------------
# Step 3 – Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(laps_df, fcy_df):
    print("\n[3/5] Engineering features …")

    # Tire age progress (normalized)
    laps_df["tire_age_progress"] = (laps_df["tireage"] / laps_df["nolaps"]).clip(0, 1)

    # Relative compound number
    laps_df["rel_compound_num"] = laps_df.apply(
        lambda r: compound_to_rel(str(r["compound"]) if pd.notna(r["compound"]) else "",
                                  r["avail_dry"]),
        axis=1
    )

    # Track category
    laps_df["location_cat"] = laps_df["location"].map(TRACK_CATEGORY).fillna(2).astype(int)

    # Position encoded (1=leader, 2=rest — as used in supervised VSE)
    laps_df["pos_encoded"] = np.where(laps_df["position"] == 1, 1, 2)

    # Cumulative pits so far and remaining pit stops
    laps_df = laps_df.sort_values(["race_id", "driver_id", "lapno"])
    pit_laps = laps_df[laps_df["is_pit"] == 1]
    total_pits = (pit_laps.groupby(["race_id", "driver_id"])
                  .size().reset_index(name="total_pits"))
    laps_df = laps_df.merge(total_pits, on=["race_id", "driver_id"], how="left")
    laps_df["total_pits"] = laps_df["total_pits"].fillna(0).astype(int)
    laps_df["pits_so_far"] = laps_df.groupby(["race_id", "driver_id"])["is_pit"].cumsum()
    laps_df["remaining_pits"] = (
        laps_df["total_pits"] - (laps_df["pits_so_far"] - laps_df["is_pit"])
    ).clip(0, 3)

    # Tire change of pursuer (did the driver directly behind pit last lap?)
    laps_df = laps_df.sort_values(["race_id", "lapno", "position"]).reset_index(drop=True)
    laps_df["pursuer_pos"] = laps_df["position"] + 1
    pursuer_lkp = laps_df[["race_id", "lapno", "position", "is_pit"]].copy()
    pursuer_lkp.columns = ["race_id", "lapno", "pursuer_pos", "tirechange_pursuer"]
    laps_df = laps_df.merge(pursuer_lkp, on=["race_id", "lapno", "pursuer_pos"], how="left")
    laps_df["tirechange_pursuer"] = laps_df["tirechange_pursuer"].fillna(0).astype(int)
    laps_df = laps_df.drop(columns=["pursuer_pos"])

    # Close ahead (driver behind is within 1.5 s)
    behind_lkp = laps_df[["race_id", "lapno", "position", "interval"]].copy()
    behind_lkp["position"] = behind_lkp["position"] - 1
    behind_lkp.columns = ["race_id", "lapno", "position", "behind_interval"]
    laps_df = laps_df.merge(behind_lkp, on=["race_id", "lapno", "position"], how="left")
    laps_df["close_ahead"] = (
        laps_df["behind_interval"].notna() & (laps_df["behind_interval"] <= 1.5)
    ).astype(int)
    laps_df = laps_df.drop(columns=["behind_interval"])

    # FCY status (0=none, 1=first VSC lap, 2=further VSC, 3=first SC, 4=further SC)
    print("    Computing FCY status …")
    fcy_rows = []
    for _, row in fcy_df.iterrows():
        ftype = str(row["type"]).strip()
        for lap in range(int(row["startlap"]), int(row["endlap"]) + 1):
            is_first = (lap == int(row["startlap"]))
            status = (3 if is_first else 4) if ftype == "SC" else (1 if is_first else 2)
            fcy_rows.append({"race_id": row["race_id"], "lapno": lap, "fcy_status": status})

    if fcy_rows:
        fcy_lap_df = (pd.DataFrame(fcy_rows)
                      .groupby(["race_id", "lapno"])["fcy_status"].max()
                      .reset_index())
        laps_df = laps_df.merge(fcy_lap_df, on=["race_id", "lapno"], how="left")
        laps_df["fcy_status"] = laps_df["fcy_status"].fillna(0).astype(int)
    else:
        laps_df["fcy_status"] = 0

    # Used 2 compounds flag (for CC)
    def mark_used_2(group):
        group = group.sort_values("lapno").copy()
        seen, flags = set(), []
        for _, r in group.iterrows():
            seen.add(r["compound"])
            flags.append(int(len(seen) >= 2))
        group["used_2_compounds"] = flags
        return group

    laps_df = laps_df.groupby(["race_id", "driver_id"],
                               group_keys=False).apply(mark_used_2)

    return laps_df


# ---------------------------------------------------------------------------
# Step 4 – Build TC and CC datasets
# ---------------------------------------------------------------------------

def build_datasets(laps_df):
    print("\n[4/5] Building TC and CC datasets …")

    # TC: all laps, result position ≤ 10, from lap 2 onward, valid compound
    tc_df = laps_df[
        (laps_df["result_position"] <= 10) &
        (laps_df["lapno"] >= 2) &
        (laps_df["rel_compound_num"] >= 0)
    ].dropna(subset=["tire_age_progress", "race_progress"]).copy()

    tc_cols = [
        "race_id", "driver_id", "lapno", "season", "location",
        "tire_age_progress", "race_progress", "pos_encoded",
        "rel_compound_num", "fcy_status", "remaining_pits",
        "tirechange_pursuer", "location_cat", "close_ahead",
        "is_pit"  # label
    ]
    tc_df = tc_df[tc_cols].reset_index(drop=True)

    # CC: pit-stop laps only, result position ≤ 15, valid next compound
    cc_df = laps_df[
        (laps_df["result_position"] <= 15) &
        (laps_df["is_pit"] == 1)
    ].dropna(subset=["nextcompound"]).copy()

    cc_df["cc_label"] = cc_df.apply(
        lambda r: compound_to_rel(
            str(r["nextcompound"]) if pd.notna(r["nextcompound"]) else "",
            r["avail_dry"]
        ), axis=1
    )
    cc_df = cc_df[cc_df["cc_label"] >= 0].copy()

    cc_cols = [
        "race_id", "driver_id", "lapno", "season", "location",
        "race_progress", "location_cat", "rel_compound_num",
        "remaining_pits", "used_2_compounds", "n_dry",
        "cc_label"  # label
    ]
    cc_df = cc_df[cc_cols].reset_index(drop=True)

    print(f"    TC dataset : {len(tc_df):,} rows  "
          f"| pit-stop laps: {tc_df['is_pit'].sum():,}  (paper: ~4,087)")
    print(f"    CC dataset : {len(cc_df):,} rows  (paper: ~2,757)")
    print(f"\n    TC class balance:")
    print(f"      No pit : {(tc_df['is_pit']==0).sum():,}  "
          f"({(tc_df['is_pit']==0).mean()*100:.1f}%)")
    print(f"      Pit    : {(tc_df['is_pit']==1).sum():,}  "
          f"({(tc_df['is_pit']==1).mean()*100:.1f}%)")
    print(f"\n    CC compound distribution (0=Hard, 1=Medium, 2=Soft):")
    print(cc_df["cc_label"].value_counts().sort_index().to_string())

    return tc_df, cc_df


# ---------------------------------------------------------------------------
# Step 5 – Save
# ---------------------------------------------------------------------------

def save(tc_df, cc_df, out_dir):
    print(f"\n[5/5] Saving CSV files to: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    tc_path = os.path.join(out_dir, "tc_features.csv")
    cc_path = os.path.join(out_dir, "cc_features.csv")
    tc_df.to_csv(tc_path, index=False)
    cc_df.to_csv(cc_path, index=False)
    print(f"    Saved tc_features.csv  ({len(tc_df):,} rows)")
    print(f"    Saved cc_features.csv  ({len(cc_df):,} rows)")
    return tc_path, cc_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"\nERROR: Database not found at '{DB_PATH}'")
        print("Usage: python preprocess_vse.py [path/to/F1_timingdata_2014_2019.sqlite]")
        sys.exit(1)

    out_dir = os.path.join(_here, "vse_features")

    laps_df, races_df, fcy_df, sf_df = load_raw(DB_PATH)
    laps_df = apply_filters(laps_df, races_df, sf_df)
    laps_df = engineer_features(laps_df, fcy_df)
    tc_df, cc_df = build_datasets(laps_df)
    save(tc_df, cc_df, out_dir)

    print("\nDone. Run evaluate_model.py next.")
