import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import re

def normalize_name(name):
    if pd.isna(name): return np.nan
    name = str(name)
    name = "".join(c for c in unicodedata.normalize('NFD', name)
                  if unicodedata.category(c) != 'Mn')
    s = name.lower().strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        s = f"{parts[1]} {parts[0]}"
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())

def clean_currency(value):
    if isinstance(value, str):
        clean_val = value.replace('$', '').replace(',', '').strip()
        try: return float(clean_val)
        except: return 0.0
    return value

# Load raw files
sc = pd.read_csv("stat_cast_two.csv")
war = pd.read_csv("war_stats.csv")
con = pd.read_csv("spotrac.csv")

# Process Keys
sc['player_key'] = sc[sc.columns[0]].apply(normalize_name)
war['player_key'] = war['Player'].apply(normalize_name)
con['player_key'] = con['Player'].apply(normalize_name)

# Clean numeric columns
con['AAV'] = con['AAV'].apply(clean_currency)
con['Start'] = pd.to_numeric(con['Start'], errors='coerce').fillna(2024).astype(int)
con['End'] = pd.to_numeric(con['End'], errors='coerce').fillna(2025).astype(int)
# Filter for 2025 season contracts
con = con[(con['Start'] <= 2025) & (con['End'] >= 2025)]
war['WAR'] = pd.to_numeric(war['WAR'], errors='coerce').fillna(0)

# Handle players with multiple teams
war_dedup = war.copy()
war_dedup['is_2tm'] = war_dedup['Team'] == '2TM'
war_dedup = war_dedup.sort_values(['player_key', 'is_2tm', 'WAR'], ascending=[True, False, False])
war_dedup = war_dedup.drop_duplicates(subset=['player_key'], keep='first')
war = war_dedup.drop('is_2tm', axis=1)

# Aggregate Statcast (Numeric only)
sc_numeric = sc.select_dtypes(include=[np.number]).copy()
sc_numeric['player_key'] = sc['player_key']
sc_agg = sc_numeric.groupby('player_key', as_index=False).mean()

# THE MERGE - Start with WAR to include all players, then add Statcast and contract data
# Make Statcast optional since it has limited coverage
df_result = war[['player_key', 'WAR', 'Team', 'Pos']].merge(sc_agg, on="player_key", how="left")
df_result = df_result.merge(con[['player_key', 'AAV', 'Pos']], on="player_key", how="left", suffixes=('', '_contract'))

df_result['Pos'] = df_result['Pos_contract'].fillna(df_result['Pos'])
df_result = df_result.drop('Pos_contract', axis=1)

# Manual overrides
overrides = {
    "jeremy pena": 9475000,
    "pete crow armstrong": 780000,
    "christian walker": 22000000
}

for player, salary in overrides.items():
    df_result.loc[df_result['player_key'] == player, 'AAV'] = salary

# Fill missing values
df_result['WAR'] = df_result['WAR'].fillna(0)
df_result['AAV'] = df_result['AAV'].fillna(0)

# Filter: Non-pitchers only (include all AAV values, including 0)
df_filtered = df_result[~df_result['Pos'].isin(['SP', 'RP'])].copy()

print(f"Total players (non-pitchers, all AAV): {len(df_filtered)}")

# Display all available columns to help user find the right ones
print("\nAvailable columns in dataset:")
print(df_filtered.columns.tolist())

# Calculate derived metrics
df_filtered['zone_swinging'] = df_filtered['z_swing_percent'] - df_filtered['oz_swing_percent']
df_filtered['zone_take_rate'] = 100 - df_filtered['z_swing_percent']
df_filtered['contact_quality'] = df_filtered['poorlytopped_percent'] + df_filtered['poorlyunder_percent']

# Column mapping for the requested metrics
column_mapping = {
    'meatball_swing_percent': 'Meatball Swing %',
    'attack_angle': 'Attack Angle',
    'zone_swinging': 'Zone Swinging',
    'zone_take_rate': 'Zone Take Rate',
    'contact_quality': 'Contact Quality',
    'avg_swing_length': 'Avg Swing Length',
    'f_strike_percent': 'F Strike %',
    'swords': 'Swords',
    'WAR': 'WAR'
}

# Check which columns exist in the dataframe
available_columns = {}
for col_key, col_label in column_mapping.items():
    if col_key in df_filtered.columns:
        available_columns[col_key] = col_label
    else:
        print(f"Warning: Column '{col_key}' not found in dataset")

print(f"\nSelected columns for correlation matrix: {list(available_columns.keys())}")

# Extract only the columns we have
if available_columns:
    correlation_data = df_filtered[list(available_columns.keys())].dropna()
    
    print(f"Rows with complete data: {len(correlation_data)}")
    
    # Show which columns have the most missing data
    print("\nMissing data analysis:")
    for col in available_columns.keys():
        missing = df_filtered[col].isnull().sum()
        total = len(df_filtered)
        print(f"{col}: {missing}/{total} missing ({missing/total*100:.1f}%)")
    
    # Calculate correlation matrix
    corr_matrix = correlation_data.corr()
    
    # Rename columns to user-friendly names
    corr_matrix.columns = [available_columns.get(col, col) for col in corr_matrix.columns]
    corr_matrix.index = [available_columns.get(idx, idx) for idx in corr_matrix.index]
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Correlation Coefficient'},
        linewidths=0.5,
        linecolor='gray',
        vmin=-1,
        vmax=1,
        square=True
    )
    
    plt.title('Hitter Performance Metrics vs. Market Value\n(Non-Pitchers, AAV > $0)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig('hitter_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✅ Correlation matrix saved as 'hitter_correlation_matrix.png'")
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
else:
    print("No matching columns found. Please check the available columns above.")
