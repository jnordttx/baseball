import pandas as pd
import numpy as np
import unicodedata
import re

def normalize_name(name):
    if pd.isna(name): return np.nan
    # Remove accents (Peña -> Pena)
    name = "".join(c for c in unicodedata.normalize('NFD', str(name)) if unicodedata.category(c) != 'Mn')
    s = name.lower().strip()
    # Handle "Last, First"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2: s = f"{parts[1]} {parts[0]}"
    # Strip punctuation and suffixes
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())

def to_numeric_money(series):
    return pd.to_numeric(series.astype(str).replace(r"[\$,]", "", regex=True).str.strip(), errors="coerce")

# 1. Load your raw CSVs
df_statcast = pd.read_csv("stat_cast_two.csv")
df_war = pd.read_csv("war_stats.csv")
df_contracts = pd.read_csv("spotrac.csv")

# 2. Prepare Keys
df_statcast['player_key'] = df_statcast[df_statcast.columns[0]].apply(normalize_name)
df_war['player_key'] = df_war['Player'].apply(normalize_name)
df_contracts['player_key'] = df_contracts['Player'].apply(normalize_name)

# 3. Clean Money Columns
for col in ["Value", "AAV"]:
    if col in df_contracts.columns:
        df_contracts[col] = to_numeric_money(df_contracts[col])

# 4. Merge Data (Prioritize WAR and Team from the WAR file)
df = df_statcast.merge(df_war[['player_key', 'WAR', 'Team']], on='player_key', how='left')
df = df.merge(df_contracts[['player_key', 'AAV']], on='player_key', how='left')

# 5. 2026 STRATEGIC OVERRIDES
# Fixing the players who were showing as 0 due to data lag or pre-arb status
overrides = {
    "jeremy pena": 9475000,
    "christian walker": 20000000,
    "wyatt langford": 820000,
    "michael busch": 780500,
    "pete crow armstrong": 820000
}

for player, salary in overrides.items():
    df.loc[df['player_key'] == player, 'AAV'] = salary

# 6. Save the finished product
df.to_csv("master_deployment_data.csv", index=False)
print("✅ Kitchen is clean! 'master_deployment_data.csv' has been created.")