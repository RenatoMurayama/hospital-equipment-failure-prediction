# Data preparation – decision log

Every decision, the evidence behind it, and the alternative that was rejected.

## 0. Data understanding

| Table | Shape | Issues found |
|---|---|---|
| `equipamentos` | 540 × 14 | 30 duplicates; inconsistent categories; impossible values; missing values; possible outliers |
| `leituras` | 1258 × 7 | non-predictive ids and dates; impossible values; missing values; 1–4 readings per device; 1 orphan id |

## 1. Duplicates

- 540 rows, but only **510 distinct `id_equipamento`**.
- The 30 extra rows are **exact copies** (every column identical), all located at the end of the file (rows 511–540). This points to an export or merge error, not to different devices.
- **Decision:** `Remove Duplicates` on `id_equipamento`, leaving 510 devices.
- **Why it matters:** duplicates weigh double in statistics and models, and a copy split across train and test leaks information.

## 2. Inconsistent categories

| Attribute | Found | Should be |
|---|---|---|
| `criticidade` | baixa 336, alta 166, Alta 3, ALTA 3, Baixa 2 | alta 172 / baixa 338 |
| `em_uso` | sim 413, nao 91, Sim 3, Nao 3 | sim 416 / nao 94 |

- **Decision:** `Map` (exact value replacement), chosen over `Replace`, which works on substrings or regular expressions and can match unintended values.
- **Root cause:** free-text entry. A dropdown in the source system would prevent it.

## 3. Impossible values → missing

Rule: **impossible** values become missing; **extreme but plausible** values are kept.

| Attribute | Value(s) | Rule | Reasoning |
|---|---|---|---|
| `ano_instalacao` | 1799, 2099 | < 1990 or > 2026 | no electromedical devices in 1799; 2099 is in the future (real range 2005–2024) |
| `horas_uso_dia` | 48 | > 24 | a day has 24 h; 0 is valid (idle device) |
| `temperatura_operacao_c` | −15 | < 0 | a running device inside a hospital cannot be below 0 °C (IEC 60601-1: typical ambient 10–40 °C); ~60 °C kept for the outlier step |
| `custo_aquisicao_eur` | −5000 | < 0 | a negative cost does not exist; 0 € kept (possible donation or loan) |
| `consumo_medio_kwh`, `custo_pecas_eur`, `custo_mao_obra_eur` | −0.58; −46, −15, −7; −80 | < 0 | negative energy use or cost does not exist; costs are summed later |
| `pressao_bar` | −2 | < 0 | every other value lies in 0.5–6.4 bar |
| `corrente_a` | −3 … −1 (9 readings) | < 0 | motor current draw is not negative |
| `ruido_db` | 1500 | > 194 | physically impossible (the loudest possible sound in air is ~194 dB); likely a decimal error |
| `rpm_motor` | 15 000 | > 10 000 | ~5× the next highest value (2930); likely an extra zero |

**Why missing and not delete the row?** The other attributes stay, including the target. Deleting would remove ~20 devices (≈4%), and it would bias the sample if the errors are not random.

## 4. Missing values

| Table | Most affected attribute | % |
|---|---|---|
| equipamentos | `consumo_medio_kwh` | 10.2% |
| leituras | `rpm_motor` | 6.0% |

| Attribute | Mean | Median | Skew | Strategy |
|---|---|---|---|---|
| `consumo_medio_kwh` | 3.31 | 3.21 | 1.48 | median |
| `temperatura_operacao_c` | 34.17 | 34.00 | 1.34 | median |
| `horas_uso_dia` | 12.05 | 11.80 | 0.01 | mean |
| `custo_aquisicao_eur` | 17 809 | 17 802 | −0.13 | mean |

- **Readings were not imputed before aggregation.** The mean per device ignores missing values, so each device's mean uses only its own real readings. The few devices with no valid reading for a sensor (5–11) were imputed after the join.

## 5. Feature engineering

`custo_manutencao_eur = custo_pecas_eur + custo_mao_obra_eur` (mean 1195 €; the two original columns were removed). This step runs after costs are cleaned and imputed, so the sum does not inherit their errors.

## 6. Normalisation

z-transformation (mean 0, sd 1) of `horas_uso_dia`, `temperatura_operacao_c`, `consumo_medio_kwh`, `custo_aquisicao_eur`, `custo_manutencao_eur` and `ano_instalacao`. Parameters are saved in `data/processed/normalization_parameters.csv` so values can be converted back to original units (e.g. z = 6.52 → 34.17 + 6.52 × 4.62 ≈ 64.3 °C).

## 7. Outliers

1. `avaria` was set as **label** and `id_equipamento` as **id** before detection, so neither enters the distance calculation. With `avaria` included, 2 failed devices were flagged as outliers.
2. **Distance-based method (top 10 by default):** the "10" is arbitrary. Only 4 of the 10 were clearly anomalous (z > 5 in temperature and energy use).
3. **LOF** (MinPts 10–20, Euclidean): sorted scores show a jump after the 4th device (AI Studio: 2.37 → 1.40), followed by a smooth decline.
4. **Decision:** remove devices with LOF > 2: **EQ0032, EQ0143, EQ0220, EQ0267**.

| ID | Type | Hours/day | Temp | Energy | In use | Failure |
|---|---|---|---|---|---|---|
| EQ0032 | Defibrillator | 22.0 | 64.3 °C | 10.1 kWh | yes | 0 |
| EQ0143 | Defibrillator | 22.2 | 59.6 °C | 11.8 kWh | **no** | 0 |
| EQ0220 | Monitor | 23.9 | 60.8 °C | 10.4 kWh | yes | 0 |
| EQ0267 | Monitor | 22.6 | 57.5 °C | 10.0 kWh | yes | 0 |

These look like record errors: defibrillators rarely run actively for 22 h/day, EQ0143 is "not in use" yet logs 22.2 h/day, and none failed. They make up 0.8% of the data.

## 8. Aggregation and join

- Readings aggregated per device (mean) and renamed to `pressao_media`, `rpm_media`, `corrente_media` and `ruido_media`. This gives 511 rows: 510 devices plus the **orphan `EQ9999`**, which exists in readings but not in the inventory.
- **Left join** (devices on the left) keeps every device and drops the orphan. Result: **506 rows, 1 per device**.
- `rpm_media` missing went from 7 to 6, because one device without RPM data was the orphan itself.

## 9. Output

`equipment_dataset_final.csv`: **506 devices × 17 attributes**, no missing values, failure rate 15.0%.
