import pandas as pd
import numpy as np
import re
import unicodedata

def normalize_name(name):
    if pd.isna(name): return np.nan
    name = str(name)
    name = ''.join(c for c in unicodedata.normalize('NFD', name)
                  if unicodedata.category(c) != 'Mn')
    s = name.lower().strip()
    if ',' in s:
        parts = [p.strip() for p in s.split(',', 1)]
        s = f'{parts[1]} {parts[0]}'
    s = re.sub(r'[^a-z\s]', ' ', s)
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', s)
    return ' '.join(s.split())

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
df_result = war[['player_key', 'WAR', 'Team', 'Pos']].merge(sc_agg, on="player_key", how="left")
df_result = df_result.merge(con[['player_key', 'AAV', 'Pos']], on="player_key", how="left", suffixes=('', '_contract'))

# Use contract position if available, otherwise use WAR position
df_result['Pos'] = df_result['Pos_contract'].fillna(df_result['Pos'])
df_result = df_result.drop('Pos_contract', axis=1)

# Fill missing values
df_result['WAR'] = df_result['WAR'].fillna(0)
df_result['AAV'] = df_result['AAV'].fillna(0)

# Filter for non-pitchers only
df_filtered = df_result[~df_result['Pos'].isin(['SP', 'RP'])].copy()

print(f"Total non-pitcher players: {len(df_filtered)}")

# Calculate position averages
position_stats = df_filtered.groupby('Pos').agg({
    'WAR': ['mean', 'median', 'count'],
    'meatball_swing_percent': ['median', lambda x: x.quantile(0.5)],  # Top 50% threshold
    'avg_swing_length': ['median', lambda x: x.quantile(0.5)],  # Top 50% threshold
    'z_swing_percent': ['mean', 'median'],
    'oz_swing_percent': ['mean', 'median']
}).round(3)

# Flatten column names
position_stats.columns = ['war_mean', 'war_median', 'player_count', 'meatball_median', 'meatball_top50', 'swing_length_median', 'swing_length_top50', 'z_swing_mean', 'z_swing_median', 'oz_swing_mean', 'oz_swing_median']
position_stats = position_stats.reset_index()

print("\nPosition Statistics:")
print(position_stats.to_string())

# Calculate league-wide meatball swing average (across all position players)
league_meatball_avg = df_filtered['meatball_swing_percent'].mean()
print(f"\nLeague-wide meatball swing % average: {league_meatball_avg:.1f}%")

# Find players who meet criteria
candidates = []

for pos in position_stats['Pos']:
    pos_data = df_filtered[df_filtered['Pos'] == pos].copy()
    if len(pos_data) < 3:  # Skip positions with too few players
        continue

    pos_stats = position_stats[position_stats['Pos'] == pos].iloc[0]

    # Criteria:
    # 1. Below average WAR for position (below median)
    below_avg_war = pos_data[pos_data['WAR'] < pos_stats['war_median']].copy()

    # 2. Above league-wide average meatball swing %
    above_avg_meatball = below_avg_war[below_avg_war['meatball_swing_percent'] > league_meatball_avg].copy()

    if not above_avg_meatball.empty:
        above_avg_meatball['position'] = pos
        above_avg_meatball['pos_war_median'] = pos_stats['war_median']
        candidates.append(above_avg_meatball)

if candidates:
    all_candidates = pd.concat(candidates)

    # Sort by WAR (lowest first) and then by meatball swing % (highest first)
    all_candidates = all_candidates.sort_values(['WAR', 'meatball_swing_percent'], ascending=[True, False])

    print(f"\nFound {len(all_candidates)} qualified players:")
    print(f"Showing top 10 (sorted by lowest WAR, then highest meatball swing %):")

    # Format the results nicely
    for idx, row in all_candidates.head(10).iterrows():
        print(f"\n{row['player_key']} ({row['Pos']}) - {row['Team']}")
        print(f"  WAR: {row['WAR']:.1f} (position median: {row['pos_war_median']:.1f})")
        print(f"  Meatball Swing %: {row['meatball_swing_percent']:.1f} (league avg: {league_meatball_avg:.1f})")
        if pd.notna(row.get('avg_swing_length')):
            print(f"  Avg Swing Length: {row['avg_swing_length']:.1f}")
        if pd.notna(row.get('zone_swinging_rate')):
            print(f"  Zone Swinging Rate: {row['zone_swinging_rate']:.1f}")
else:
    print("\nNo players found meeting all criteria.")