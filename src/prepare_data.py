"""
Hospital equipment failure - Data Preparation pipeline (CRISP-DM, phase 3).

Python reproduction of the Altair AI Studio process built for the course
"Aprendizagem e Extração do Conhecimento" (MSc Biomedical Engineering -
Medical Informatics, University of Minho).

Input : data/raw/equipamentos.csv, data/raw/leituras.csv
Output: data/processed/equipment_dataset_final.csv
        data/processed/preparation_log.csv
        figures/*.png

Run   : python src/prepare_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, FIG = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

LOG = []


def log(step, action, detail):
    LOG.append({"step": step, "action": action, "detail": detail})
    print(f"[{step}] {action}: {detail}")


# ---------------------------------------------------------------- load
eq = pd.read_csv(RAW / "equipamentos.csv")
rd = pd.read_csv(RAW / "leituras.csv")
log("00", "load", f"equipamentos {eq.shape}, leituras {rd.shape}")

# ---------------------------------------------------------------- step 01
n_before = len(eq)
eq = eq.drop_duplicates(subset="id_equipamento", keep="first").reset_index(drop=True)
log("01", "remove duplicates", f"{n_before - len(eq)} duplicated rows removed -> {len(eq)} unique devices")

rd = rd.drop(columns=["id_leitura", "data_leitura"])
log("01", "drop columns (readings)", "id_leitura, data_leitura (no predictive value at device level)")

# ---------------------------------------------------------------- step 02
for col in ["criticidade", "em_uso"]:
    before = eq[col].nunique()
    eq[col] = eq[col].str.strip().str.lower()
    log("02", f"normalise categories ({col})", f"{before} -> {eq[col].nunique()} distinct values {sorted(eq[col].unique())}")

# ---------------------------------------------------------------- step 03
IMPOSSIBLE_EQ = {
    "ano_instalacao": lambda s: (s < 1990) | (s > 2026),
    "horas_uso_dia": lambda s: s > 24,
    "temperatura_operacao_c": lambda s: s < 0,
    "custo_aquisicao_eur": lambda s: s < 0,
    "consumo_medio_kwh": lambda s: s < 0,
    "custo_pecas_eur": lambda s: s < 0,
    "custo_mao_obra_eur": lambda s: s < 0,
}
IMPOSSIBLE_RD = {
    "pressao_bar": lambda s: s < 0,
    "corrente_a": lambda s: s < 0,
    "ruido_db": lambda s: s > 194,      # loudest possible sound in air ~194 dB
    "rpm_motor": lambda s: s > 10_000,  # ~5x the next highest value
}
for df, rules, name in [(eq, IMPOSSIBLE_EQ, "equipamentos"), (rd, IMPOSSIBLE_RD, "leituras")]:
    for col, rule in rules.items():
        mask = rule(df[col])
        df.loc[mask, col] = np.nan
        log("03", f"impossible -> missing ({name}.{col})", f"{int(mask.sum())} values")

# ---------------------------------------------------------------- step 04
missing_pct = pd.concat([
    (eq.isna().mean() * 100).rename("pct").to_frame().assign(table="equipamentos"),
    (rd.isna().mean() * 100).rename("pct").to_frame().assign(table="leituras"),
]).query("pct > 0").sort_values(["table", "pct"], ascending=[True, False])

for col in ["consumo_medio_kwh", "temperatura_operacao_c"]:   # right-skewed -> median
    med = eq[col].median()
    eq[col] = eq[col].fillna(med)
    log("04", f"impute median ({col})", f"skew={eq[col].skew():.2f}, median={med:.2f}")
num_eq = ["ano_instalacao", "horas_uso_dia", "custo_aquisicao_eur", "custo_pecas_eur", "custo_mao_obra_eur"]
eq[num_eq] = eq[num_eq].fillna(eq[num_eq].mean())
log("04", "impute mean (symmetric attributes)", ", ".join(num_eq))
log("04", "readings", "not imputed before aggregation (mean per device ignores missing values)")

# ---------------------------------------------------------------- step 05
eq["custo_manutencao_eur"] = eq["custo_pecas_eur"] + eq["custo_mao_obra_eur"]
eq = eq.drop(columns=["custo_pecas_eur", "custo_mao_obra_eur"])
log("05", "feature engineering", f"custo_manutencao_eur = parts + labour (mean {eq['custo_manutencao_eur'].mean():.2f} EUR)")

# ---------------------------------------------------------------- step 06
NORM = ["horas_uso_dia", "temperatura_operacao_c", "consumo_medio_kwh",
        "custo_aquisicao_eur", "custo_manutencao_eur", "ano_instalacao"]
norm_params = eq[NORM].agg(["mean", "std"]).T
eq[NORM] = (eq[NORM] - eq[NORM].mean()) / eq[NORM].std()
norm_params.to_csv(OUT / "normalization_parameters.csv")
log("06", "z-transformation", "mean 0 / std 1 for " + ", ".join(NORM))

# ---------------------------------------------------------------- step 07
# LOF on the normalised numeric attributes only (label `avaria` and id excluded).
# Like AI Studio's LOF with a MinPts range, the score is the max over k = 10..20.
X = eq[NORM].to_numpy()
lof = np.max([-LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_ for k in range(10, 21)], axis=0)
eq["lof_score"] = lof
LOF_THRESHOLD = 2.0
outliers = eq.loc[eq["lof_score"] > LOF_THRESHOLD, "id_equipamento"].tolist()
lof_sorted = np.sort(lof)[::-1]
eq = eq[eq["lof_score"] <= LOF_THRESHOLD].drop(columns="lof_score").reset_index(drop=True)
log("07", "LOF outliers", f"threshold {LOF_THRESHOLD}: removed {outliers} -> {len(eq)} devices")

# ---------------------------------------------------------------- step 08
agg = (rd.groupby("id_equipamento")[["pressao_bar", "rpm_motor", "corrente_a", "ruido_db"]]
         .mean()
         .rename(columns={"pressao_bar": "pressao_media", "rpm_motor": "rpm_media",
                          "corrente_a": "corrente_media", "ruido_db": "ruido_media"})
         .reset_index())
orphans = sorted(set(agg["id_equipamento"]) - set(pd.read_csv(RAW / "equipamentos.csv")["id_equipamento"]))
log("08", "aggregate readings", f"{len(agg)} rows; orphan ids: {orphans}")

final = eq.merge(agg, on="id_equipamento", how="left")
sensor_cols = ["pressao_media", "rpm_media", "corrente_media", "ruido_media"]
log("08", "left join", f"{len(final)} rows, missing sensor means: {final[sensor_cols].isna().sum().to_dict()}")
final[sensor_cols] = final[sensor_cols].fillna(final[sensor_cols].mean())

# ---------------------------------------------------------------- step 09
final.to_csv(OUT / "equipment_dataset_final.csv", index=False)
pd.DataFrame(LOG).to_csv(OUT / "preparation_log.csv", index=False)
assert final["id_equipamento"].is_unique and final.isna().sum().sum() == 0
log("09", "export", f"{final.shape[0]} devices x {final.shape[1]} attributes, 0 missing, failure rate {final['avaria'].mean():.1%}")

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titleweight": "bold", "axes.titlesize": 13})


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# 1. missing values
fig, ax = plt.subplots(figsize=(9, 5), dpi=150); fig.patch.set_facecolor(SURF); style(ax)
labels = [f"{t}.{c}" for c, t in zip(missing_pct.index, missing_pct["table"])]
colors = [BLUE if t == "equipamentos" else ORANGE for t in missing_pct["table"]]
ax.barh(labels[::-1], missing_pct["pct"][::-1], color=colors[::-1], height=0.6)
for y, v in enumerate(missing_pct["pct"][::-1]):
    ax.text(v + 0.15, y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)
ax.set_xlabel("% missing (after impossible values were flagged)")
ax.set_title("Missing values per attribute", loc="left")
plt.tight_layout(); plt.savefig(FIG / "01_missing_values.png", facecolor=SURF); plt.close()

# 2. LOF scores
fig, ax = plt.subplots(figsize=(9, 5), dpi=150); fig.patch.set_facecolor(SURF); style(ax)
top = lof_sorted[:40]
ax.bar(range(1, len(top) + 1), top, color=[RED if v > LOF_THRESHOLD else BLUE for v in top], width=0.7)
ax.axhline(LOF_THRESHOLD, color=INK2, ls="--", lw=1.2)
ax.text(len(top), LOF_THRESHOLD + 0.06, f"threshold LOF > {LOF_THRESHOLD:g}", ha="right", color=INK2, fontsize=10)
ax.set_xlabel("Devices ranked by LOF score (top 40 of 510)")
ax.set_ylabel("LOF score")
ax.set_title("Data-driven outlier cut: the jump after the 4th device", loc="left")
plt.tight_layout(); plt.savefig(FIG / "02_lof_scores.png", facecolor=SURF); plt.close()

# 3. failure rate by categorical attribute
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=150, sharey=True); fig.patch.set_facecolor(SURF)
for ax, col in zip(axes, ["tipo", "fabricante", "criticidade"]):
    style(ax)
    r = final.groupby(col)["avaria"].mean().sort_values(ascending=False) * 100
    ax.bar(r.index, r.values, color=BLUE, width=0.6)
    for x, v in enumerate(r.values):
        ax.text(x, v + 0.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_title(col, loc="left", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
axes[0].set_ylabel("failure rate (%)")
fig.suptitle("Failure rate by device type, manufacturer and criticality", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "03_failure_rate_by_category.png", facecolor=SURF); plt.close()

print("Figures saved to", FIG)
