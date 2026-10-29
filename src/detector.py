"""Oil well anomaly detection with Isolation Forest.

Pipeline:
    generate (or load) -> clean -> engineer features -> fit Isolation Forest
    -> explain flagged rows -> plot -> export CSV report

Run from the project root:
    python src/detector.py --generate        # create data/sample_data.csv, then analyse it
    python src/detector.py                   # analyse existing data/sample_data.csv
    python src/detector.py --input my.csv    # analyse your own measurements
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_data.csv"
OUTPUT_DIR = ROOT / "outputs"

METRICS = [
    "oil_rate_bpd",          # oil production, barrels/day
    "downhole_pressure_bar",  # permanent downhole gauge (3W: P-PDG)
    "wellhead_pressure_bar",  # temperature/pressure transducer (3W: P-TPT)
    "wellhead_temp_c",        # (3W: T-TPT)
    "water_cut_pct",          # basic sediment & water (3W event 1: abrupt BSW increase)
]

# Physically plausible ranges; anything outside is treated as a sensor error.
VALID_RANGES = {
    "oil_rate_bpd": (0, 10_000),
    "downhole_pressure_bar": (0, 600),
    "wellhead_pressure_bar": (0, 300),
    "wellhead_temp_c": (-20, 150),
    "water_cut_pct": (0, 100),
}

ROLL_WINDOW = 6  # hours
BASELINE_WINDOW = 7 * 24  # hours

# Colours (light chart theme)
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SERIES, ANOMALY, EVENT = "#2a78d6", "#d03b3b", "#fab219"


# --------------------------------------------------------------------------- #
# 1. Synthetic data
# --------------------------------------------------------------------------- #
WELLS = {
    # well_id: base oil rate, base water cut, downhole p, wellhead p, wellhead T
    "WELL-D": dict(q0=640, wc0=55, pdg0=172, ptp0=26, t0=57),
    "WELL-E": dict(q0=1850, wc0=9, pdg0=248, ptp0=49, t0=77),
    "WELL-F": dict(q0=1020, wc0=33, pdg0=205, ptp0=35, t0=65),
    "WELL-G": dict(q0=1380, wc0=18, pdg0=226, ptp0=41, t0=70),
}

# Undesirable events modelled on 3W classes 4-7: (well, type, start day, start hour, duration h)
EVENTS = [
    ("WELL-D", "flow_instability", 27, 8, 48),
    ("WELL-D", "choke_restriction", 88, 15, 8),
    ("WELL-D", "choke_scaling", 140, 0, 96),
    ("WELL-E", "productivity_loss", 45, 12, 60),
    ("WELL-E", "choke_restriction", 121, 4, 6),
    ("WELL-F", "choke_scaling", 33, 6, 120),
    ("WELL-F", "flow_instability", 102, 18, 36),
    ("WELL-F", "productivity_loss", 157, 2, 48),
    ("WELL-G", "choke_restriction", 19, 22, 10),
    ("WELL-G", "flow_instability", 74, 5, 40),
]


def generate_synthetic_data(days: int = 180, seed: int = 7) -> pd.DataFrame:
    """Hourly measurements for four wells with injected events and dirty data.

    The ``event_type`` column is ground truth for evaluation only - the model never sees it.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-06-01")
    hours = days * 24
    ts = pd.date_range(start, periods=hours, freq="h")
    t_days = np.arange(hours) / 24
    frames = []

    for well, p in WELLS.items():
        decline = np.exp(-0.0009 * t_days)  # natural reservoir decline
        q = p["q0"] * decline * (1 + rng.normal(0, 0.015, hours))
        wc = p["wc0"] + 0.03 * t_days + rng.normal(0, 0.6, hours)
        pdg = p["pdg0"] - 0.05 * t_days + rng.normal(0, 0.8, hours)
        ptp = p["ptp0"] * (0.97 + 0.03 * decline) + rng.normal(0, 0.4, hours)
        ambient = 2.0 * np.sin(2 * np.pi * (t_days - 0.375))  # daily cycle
        temp = p["t0"] + 6 * (q / p["q0"] - 1) + ambient + rng.normal(0, 0.35, hours)
        label = np.array(["normal"] * hours, dtype=object)

        for ev_well, kind, day, hour, dur in EVENTS:
            if ev_well != well:
                continue
            i0 = day * 24 + hour
            sl = slice(i0, i0 + dur)
            k = np.arange(dur)
            ramp = np.clip(k / 4, 0, 1)  # 4 h onset
            label[sl] = kind

            if kind == "flow_instability":
                # irregular, moderate oscillations (weaker than severe slugging)
                osc = rng.normal(0, 1, dur) * 0.6 + np.sin(2 * np.pi * k / 6)
                ptp[sl] += 3.5 * osc * ramp
                q[sl] *= 1 + 0.12 * osc * ramp
            elif kind == "productivity_loss":
                grow = k / dur
                q[sl] *= 1 - 0.35 * grow
                pdg[sl] -= 14 * grow
                ptp[sl] -= 4 * grow
                temp[sl] -= 3 * grow
            elif kind == "choke_restriction":
                step = np.minimum(k / 1.0, 1)
                ptp[sl] += 12 * step  # upstream of the choke
                q[sl] *= 1 - 0.3 * step
                pdg[sl] += 5 * step
                temp[sl] -= 2 * step
            elif kind == "choke_scaling":
                # slow build-up over days: the hardest case for a 7-day baseline
                grow = k / dur
                ptp[sl] += 9 * grow
                q[sl] *= 1 - 0.18 * grow
                pdg[sl] += 3 * grow

        frames.append(pd.DataFrame({
            "timestamp": ts,
            "well_id": well,
            "oil_rate_bpd": q.round(1),
            "downhole_pressure_bar": pdg.round(2),
            "wellhead_pressure_bar": ptp.round(2),
            "wellhead_temp_c": temp.round(2),
            "water_cut_pct": np.clip(wc, 0, 100).round(2),
            "event_type": label,
        }))

    df = pd.concat(frames, ignore_index=True)
    return _make_dirty(df, rng)


def _make_dirty(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the problems real SCADA exports have: gaps, glitches, duplicates."""
    df = df.copy()
    n = len(df)
    for col in METRICS:
        df.loc[rng.choice(n, int(0.006 * n), replace=False), col] = np.nan
    glitch = rng.choice(n, 12, replace=False)
    df.loc[glitch[:4], "wellhead_pressure_bar"] = 9999.0  # transmitter overflow
    df.loc[glitch[4:8], "oil_rate_bpd"] = -1.0            # meter error code
    df.loc[glitch[8:], "water_cut_pct"] = 120.0           # analyser out of range
    dups = df.sample(25, random_state=1)
    df = pd.concat([df, dups]).sample(frac=1, random_state=2)  # shuffle row order
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Cleaning
# --------------------------------------------------------------------------- #
def clean_data(raw: pd.DataFrame, max_gap_hours: int = 3) -> tuple[pd.DataFrame, dict]:
    """Deduplicate, sort, null out impossible values, interpolate short gaps."""
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    stats = {"rows_in": len(df)}

    df = df.drop_duplicates(subset=["well_id", "timestamp"])
    stats["duplicates_removed"] = stats["rows_in"] - len(df)
    df = df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)

    metrics = [m for m in METRICS if m in df.columns]
    stats["missing_before"] = int(df[metrics].isna().sum().sum())
    out_of_range = 0
    for col in metrics:
        lo, hi = VALID_RANGES[col]
        bad = (df[col] < lo) | (df[col] > hi)
        out_of_range += int(bad.sum())
        df.loc[bad, col] = np.nan
    stats["out_of_range_nulled"] = out_of_range

    df[metrics] = (
        df.set_index("timestamp")
        .groupby("well_id")[metrics]
        .transform(lambda s: s.interpolate(method="time", limit=max_gap_hours))
        .to_numpy()
    )
    before = len(df)
    df = df.dropna(subset=metrics).reset_index(drop=True)
    stats["rows_dropped_long_gaps"] = before - len(df)
    stats["rows_out"] = len(df)
    return df, stats


def describe_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-well summary of each metric."""
    metrics = [m for m in METRICS if m in df.columns]
    return df.groupby("well_id")[metrics].agg(["mean", "std", "min", "max"]).round(2)


# --------------------------------------------------------------------------- #
# 3. Features and model
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-well robust z-scores of level, 1 h change and 6 h volatility.

    Level is measured against the trailing 7-day median, so slow reservoir decline
    is not mistaken for an anomaly. Scaling per well matters: 900 bpd is normal
    for one well and a collapse for another.
    """
    metrics = [m for m in METRICS if m in df.columns]
    feats = {}
    g = df.groupby("well_id")
    for col in metrics:
        baseline = g[col].transform(lambda s: s.rolling(BASELINE_WINDOW, min_periods=1).median().shift(1))
        feats[f"{col}__level"] = (df[col] - baseline).fillna(0)
        feats[f"{col}__diff"] = g[col].diff().fillna(0)
        feats[f"{col}__vol"] = g[col].transform(
            lambda s: s.rolling(ROLL_WINDOW, min_periods=2).std().bfill()
        )
    X = pd.DataFrame(feats, index=df.index)
    X["well_id"] = df["well_id"]

    def robust_z(s: pd.Series) -> pd.Series:
        med = s.median()
        iqr = s.quantile(0.75) - s.quantile(0.25)
        return (s - med) / (iqr / 1.349 if iqr > 0 else s.std() or 1.0)

    return X.groupby("well_id").transform(robust_z)


def fit_detector(X: pd.DataFrame, contamination: float = 0.025, seed: int = 42) -> IsolationForest:
    model = IsolationForest(
        n_estimators=300, contamination=contamination, max_samples=512, random_state=seed
    )
    model.fit(X)
    return model


def explain(X: pd.DataFrame, top: int = 3) -> pd.Series:
    """Name the features that deviate most from the well's normal behaviour."""
    labels = {"level": "", "diff": " change", "vol": " volatility"}

    def row_reason(row: pd.Series) -> str:
        parts = []
        for name, z in row.abs().nlargest(top).items():
            metric, kind = name.split("__")
            arrow = "↑" if row[name] > 0 else "↓"
            parts.append(f"{metric}{labels[kind]} {arrow} (z={row[name]:+.1f})")
        return "; ".join(parts)

    return X.apply(row_reason, axis=1)


def detect(df: pd.DataFrame, contamination: float = 0.025) -> pd.DataFrame:
    X = build_features(df)
    model = fit_detector(X, contamination)
    out = df.copy()
    # Higher = more anomalous (sklearn's score_samples is the opposite sign).
    out["anomaly_score"] = (-model.score_samples(X)).round(4)
    out["is_anomaly"] = model.predict(X) == -1
    out["reason"] = ""
    out.loc[out["is_anomaly"], "reason"] = explain(X[out["is_anomaly"]])
    return out


# --------------------------------------------------------------------------- #
# 4. Reporting
# --------------------------------------------------------------------------- #
def group_episodes(result: pd.DataFrame, max_gap_hours: int = 2) -> pd.DataFrame:
    """Merge flagged hours that are close together into one episode to investigate."""
    flagged = result[result["is_anomaly"]].sort_values(["well_id", "timestamp"]).copy()
    gap = flagged.groupby("well_id")["timestamp"].diff() > pd.Timedelta(hours=max_gap_hours)
    new_well = flagged["well_id"] != flagged["well_id"].shift()
    flagged["episode_id"] = (gap | new_well).cumsum()
    return flagged


def build_report(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    flagged = group_episodes(result)
    hi = result.loc[result["is_anomaly"], "anomaly_score"].quantile(0.5)
    flagged["priority"] = np.where(flagged["anomaly_score"] >= hi, "high", "medium")

    cols = ["episode_id", "priority", "timestamp", "well_id", *[m for m in METRICS if m in result],
            "anomaly_score", "reason"]
    observations = flagged[cols].sort_values("anomaly_score", ascending=False)

    episodes = (
        flagged.groupby("episode_id")
        .agg(well_id=("well_id", "first"), start=("timestamp", "min"), end=("timestamp", "max"),
             flagged_hours=("timestamp", "size"), max_score=("anomaly_score", "max"),
             top_reason=("reason", lambda r: r.loc[flagged.loc[r.index, "anomaly_score"].idxmax()]))
        .sort_values("max_score", ascending=False)
        .reset_index()
    )
    return observations, episodes


def evaluate(result: pd.DataFrame) -> dict | None:
    """Compare against ground truth when the data has an ``event_type`` column."""
    if "event_type" not in result:
        return None
    truth = result["event_type"] != "normal"
    pred = result["is_anomaly"]
    tp = int((truth & pred).sum())
    events = result[truth].groupby(["well_id", "event_type", (~truth).cumsum()[truth]])
    caught = sum(g["is_anomaly"].any() for _, g in events)
    return {
        "precision": tp / max(pred.sum(), 1),
        "recall": tp / max(truth.sum(), 1),
        "events_detected": f"{caught}/{events.ngroups}",
    }


def plot_results(result: pd.DataFrame, path: Path) -> None:
    metrics = [m for m in METRICS if m in result]
    wells = sorted(result["well_id"].unique())
    titles = {
        "oil_rate_bpd": "Oil rate (bbl/d)",
        "downhole_pressure_bar": "Downhole pressure (bar)",
        "wellhead_pressure_bar": "Wellhead pressure (bar)",
        "wellhead_temp_c": "Wellhead temp (°C)",
        "water_cut_pct": "Water cut (%)",
    }
    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": GRID,
    })
    fig, axes = plt.subplots(len(metrics), len(wells), figsize=(6 * len(wells), 2.2 * len(metrics)),
                             sharex=True, squeeze=False, facecolor=SURFACE)

    for c, well in enumerate(wells):
        w = result[result["well_id"] == well]
        anom = w[w["is_anomaly"]]
        spans = []
        if "event_type" in w:
            ev = w["event_type"] != "normal"
            for _, g in w[ev].groupby((~ev).cumsum()[ev]):
                spans.append((g["timestamp"].min(), g["timestamp"].max()))
        for r, m in enumerate(metrics):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            for s, e in spans:
                ax.axvspan(s, e, color=EVENT, alpha=0.22, lw=0)
            ax.plot(w["timestamp"], w[m], color=SERIES, lw=0.8)
            ax.scatter(anom["timestamp"], anom[m], s=14, color=ANOMALY, edgecolor=SURFACE,
                       linewidth=0.6, zorder=3)
            ax.grid(axis="y", color=GRID, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if c == 0:
                ax.set_ylabel(titles.get(m, m), fontsize=8.5)
            if r == 0:
                ax.set_title(f"{well}  ·  {len(anom)} flagged hours", fontsize=10.5, loc="left")
        axes[-1, c].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    handles = [
        plt.Line2D([], [], color=SERIES, lw=1.5, label="Normal measurement"),
        plt.Line2D([], [], marker="o", ls="", color=ANOMALY, markersize=6, label="Suspicious (Isolation Forest)"),
    ]
    if "event_type" in result:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=EVENT, alpha=0.35, label="Injected event (ground truth)"))
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("Oil well anomaly detection", x=0.99, y=0.995, ha="right", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(input_path: Path, output_dir: Path, contamination: float) -> dict:
    raw = pd.read_csv(input_path)
    df, clean_stats = clean_data(raw)
    result = detect(df, contamination)
    observations, episodes = build_report(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output_dir / "suspicious_observations.csv", index=False)
    episodes.to_csv(output_dir / "anomaly_episodes.csv", index=False)
    plot_results(result, output_dir / "anomaly_plot.png")
    return {"cleaning": clean_stats, "evaluation": evaluate(result),
            "flagged": len(observations), "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--contamination", type=float, default=0.025,
                        help="expected share of anomalous rows (default 0.025)")
    parser.add_argument("--generate", action="store_true", help="regenerate synthetic data first")
    args = parser.parse_args()

    if args.generate:
        args.input.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_data().to_csv(args.input, index=False)
        print(f"Synthetic data written to {args.input}")

    summary = run(args.input, args.output_dir, args.contamination)
    print("Cleaning:", summary["cleaning"])
    if summary["evaluation"]:
        e = summary["evaluation"]
        print(f"Precision {e['precision']:.2f} | Recall {e['recall']:.2f} | "
              f"Events detected {e['events_detected']}")
    print(f"{summary['flagged']} suspicious rows in {len(summary['episodes'])} episodes")
    print(summary["episodes"].head(10).to_string(index=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
