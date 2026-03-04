import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ----------------------------
# Load data
# ----------------------------
df_statcast = pd.read_csv("stat_cast_two.csv")
df_war = pd.read_csv("war_stats.csv")
df_contracts = pd.read_csv("mlb_contracts.csv")

# ----------------------------
# Configuration
# ----------------------------
PLAYER_COL_STATCAST = df_statcast.columns[0]  # e.g., "last_name, first_name"
PLAYER_COL_WAR = "Player"
PLAYER_COL_CONTRACTS = "Player"

# Controllable / actionable inputs grouped by category
decision_skill = ["oz_swing_percent", "z_swing_percent", "meatball_swing_percent", "f_strike_percent"]
contact_ability = ["whiff_percent", "iz_contact_percent", "swords"]
mechanics = ["attack_angle", "vertical_swing_path", "attack_direction"]
contact_optimization = ["ideal_angle_rate", "sweet_spot_percent", "poorlytopped_percent", "poorlyunder_percent"]
approach = ["pull_percent", "swing_percent"]
defense = [
    "n_outs_above_average",
    "rel_league_reaction_distance",
    "rel_league_burst_distance",
    "rel_league_routing_distance",
    "rel_league_bootup_distance",
]
baserunning = ["sprint_speed", "hp_to_1b", "n_bolts"]

BASE_INPUTS = (
    decision_skill
    + contact_ability
    + mechanics
    + contact_optimization
    + approach
    + defense
    + baserunning
)

ADVANCED_FEATURES = [
    "decision_quality",             # z_swing - oz_swing
    "damage_on_good_decisions",     # meatball_swing * ideal_angle_rate
    "good_defender_index",          # composite defensive score
]

# ----------------------------
# Helper functions
# ----------------------------
def normalize_name(name):
    if pd.isna(name):
        return np.nan
    s = str(name).strip()

    # Convert "Last, First" -> "First Last"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2:
            s = f"{parts[1]} {parts[0]}"

    s = s.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)  # suffixes
    s = re.sub(r"[^a-z\s]", "", s)                  # punctuation
    s = re.sub(r"\s+", " ", s).strip()              # whitespace
    return s


def to_numeric_percent(series):
    """
    Converts strings like '32.1%' to 32.1; coerces invalid values to NaN.
    Safe for non-% numeric strings too.
    """
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    )


def to_numeric_money(series):
    """
    Converts strings like '$12,345,678' to 12345678; coerces invalid values to NaN.
    """
    return pd.to_numeric(
        series.astype(str).replace(r"[\$,]", "", regex=True).str.strip(),
        errors="coerce"
    )


def zscore(series):
    """Standardize to mean 0, std 1 (ignores NaNs)."""
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * np.nan  # all NaN if no variance
    return (s - s.mean()) / std

# ----------------------------
# Normalize player names
# ----------------------------
df_statcast["player_key"] = df_statcast[PLAYER_COL_STATCAST].apply(normalize_name)
df_war["player_key"] = df_war[PLAYER_COL_WAR].apply(normalize_name)
df_contracts["player_key"] = df_contracts[PLAYER_COL_CONTRACTS].apply(normalize_name)

# ----------------------------
# Keep only metrics that exist (prevents KeyError)
# ----------------------------
missing_base = [c for c in BASE_INPUTS if c not in df_statcast.columns]
present_base = [c for c in BASE_INPUTS if c in df_statcast.columns]

if missing_base:
    print("\nWARNING: These base input columns were NOT found in stat_cast_two.csv and will be skipped:")
    print(missing_base)

if not present_base:
    raise ValueError("None of the selected input columns were found in stat_cast_two.csv.")

# ----------------------------
# Clean numeric columns in statcast
# ----------------------------
for col in present_base:
    df_statcast[col] = to_numeric_percent(df_statcast[col])

# WAR numeric (DON'T fill missing with 0)
df_war["WAR"] = pd.to_numeric(df_war["WAR"], errors="coerce")

# Contracts money numeric (DON'T fill missing with 0)
for col in ["Value", "AAV", "Signing Bonus"]:
    if col in df_contracts.columns:
        df_contracts[col] = to_numeric_money(df_contracts[col])

# ----------------------------
# Create advanced features (row-level, then we'll aggregate)
# ----------------------------

# 1) Decision Quality = Z-Swing - O-Swing
if "z_swing_percent" in df_statcast.columns and "oz_swing_percent" in df_statcast.columns:
    df_statcast["decision_quality"] = (
        pd.to_numeric(df_statcast["z_swing_percent"], errors="coerce")
        - pd.to_numeric(df_statcast["oz_swing_percent"], errors="coerce")
    )

# 2) Damage on Good Decisions = meatball_swing% * ideal_angle_rate
if "meatball_swing_percent" in df_statcast.columns and "ideal_angle_rate" in df_statcast.columns:
    df_statcast["damage_on_good_decisions"] = (
        pd.to_numeric(df_statcast["meatball_swing_percent"], errors="coerce")
        * pd.to_numeric(df_statcast["ideal_angle_rate"], errors="coerce")
    )

# 3) Good Defender Index (composite)
#    We standardize and combine: OAA (+) + reaction (+) + burst (+) + routing (+) + bootup (+)
#    NOTE: If any of these are "lower is better" in your source, flip the sign here.
def_cols = [
    "n_outs_above_average",
    "rel_league_reaction_distance",
    "rel_league_burst_distance",
    "rel_league_routing_distance",
    "rel_league_bootup_distance",
]
present_def_cols = [c for c in def_cols if c in df_statcast.columns]
if present_def_cols:
    # Create a row-wise index using z-scores of each defensive metric (computed across all rows)
    zdef = []
    for c in present_def_cols:
        zdef.append(zscore(df_statcast[c]))
    df_statcast["good_defender_index"] = np.nanmean(np.vstack([z.to_numpy() for z in zdef]), axis=0)

# ----------------------------
# Aggregate to player level (mean)
# NOTE: If you want season-level, include "year" in groupby and merge on (player_key, year)
# ----------------------------
ALL_FEATURES_TO_AGG = list(dict.fromkeys(present_base + [c for c in ADVANCED_FEATURES if c in df_statcast.columns]))

statcast_player = (
    df_statcast
    .dropna(subset=["player_key"])
    .groupby("player_key", as_index=False)[ALL_FEATURES_TO_AGG]
    .mean()
)

# ----------------------------
# Reduce WAR dataset to one row per player (highest WAR row kept)
# ----------------------------
war_player = (
    df_war
    .dropna(subset=["player_key"])
    .sort_values("WAR", ascending=False)
    .drop_duplicates("player_key")[["player_key", "WAR"]]
)

# ----------------------------
# Reduce contracts dataset to one row per player (highest AAV row kept)
# ----------------------------
contracts_player = (
    df_contracts
    .dropna(subset=["player_key"])
    .sort_values("AAV", ascending=False)
    .drop_duplicates("player_key")[["player_key", "AAV", "Value", "Start", "End", "Yrs", "Signing Bonus"]]
)

# ----------------------------
# Merge datasets
# ----------------------------
df = (
    statcast_player
    .merge(war_player, on="player_key", how="left")
    .merge(contracts_player, on="player_key", how="left")
)

# ----------------------------
# Diagnostics
# ----------------------------
print("\nTotal players (post-merge):", len(df))
print("WAR match rate:", f"{df['WAR'].notna().mean():.1%}")
print("AAV match rate:", f"{df['AAV'].notna().mean():.1%}")

print("\nTop 10 WAR players (matched):")
print(
    df.dropna(subset=["WAR"])
      .sort_values("WAR", ascending=False)
      .head(10)[["player_key", "WAR", "AAV"]]
)

# ----------------------------
# Build correlation dataset (require WAR and AAV)
# ----------------------------
feature_cols = [c for c in ALL_FEATURES_TO_AGG if c in df.columns]
corr_cols = feature_cols + ["WAR", "AAV"]

df_corr = df[corr_cols].dropna(subset=["WAR", "AAV"])
df_corr = df_corr.select_dtypes(include=[np.number])

# ----------------------------
# Rank features by correlation with WAR (abs), pick Top 10
# ----------------------------
war_corr_all = (
    df_corr.corr()
    .loc[feature_cols, "WAR"]
    .dropna()
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

top10_features = war_corr_all.head(10).index.tolist()

print("\nTop 10 features by absolute correlation with WAR (from your controllable + advanced set):")
print(pd.DataFrame({
    "corr_with_WAR": war_corr_all.head(10),
    "corr_with_AAV": df_corr.corr().loc[top10_features, "AAV"]
}).to_string(float_format=lambda x: f"{x:0.3f}"))

# ----------------------------
# Trim dataset to Top 10 features + advanced features + WAR + AAV
# (ensures decision_quality / damage_on_good_decisions / good_defender_index are included)
# ----------------------------
must_keep = [c for c in ADVANCED_FEATURES if c in df_corr.columns]
final_features = list(dict.fromkeys(top10_features + must_keep))  # keep order, no duplicates

final_cols = final_features + ["WAR", "AAV"]
df_final = df_corr[final_cols].copy()

# ----------------------------
# Correlation matrix on the trimmed dataset
# ----------------------------
corr_final = df_final.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_final,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Trimmed Strategy Matrix: Top 10 WAR-Correlated Inputs + Advanced Features (with AAV)", fontsize=13)
plt.tight_layout()
plt.show()

# ----------------------------
# Print the final Top correlations table again (clean)
# ----------------------------
final_summary = pd.DataFrame({
    "corr_with_WAR": corr_final.loc[final_features, "WAR"],
    "corr_with_AAV": corr_final.loc[final_features, "AAV"]
}).sort_values(by="corr_with_WAR", key=lambda s: s.abs(), ascending=False)

print("\nFinal feature set (Top 10 by WAR correlation + advanced features) with correlations to WAR and AAV:")
print(final_summary.to_string(float_format=lambda x: f"{x:0.3f}"))

from sklearn.linear_model import LinearRegression

# 1. Filter for players who actually have a contract (AAV) and enough data
df_market = df.dropna(subset=['AAV'] + final_features).copy()

# 2. Define Features (X) and Target (y)
X = df_market[final_features]
y = df_market['AAV']

# 3. Fit the Market Model
model = LinearRegression()
model.fit(X, y)

# 4. Generate "Fair Market Value" (FMV) and Residuals
df_market['Predicted_AAV'] = model.predict(X)
df_market['Market_Surplus'] = df_market['Predicted_AAV'] - df_market['AAV']

# 5. Identify the Top 10 "Value Deployments"
arbitrage_targets = df_market.sort_values(by='Market_Surplus', ascending=False).head(10)

print("\n--- TOP ARBITRAGE TARGETS (Undervalued based on Statcast) ---")
print(arbitrage_targets[['player_key', 'AAV', 'Predicted_AAV', 'Market_Surplus']])