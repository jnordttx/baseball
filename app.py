import streamlit as st
import pandas as pd
import numpy as np
import re

# --- 1. SETTINGS & THEME ---
st.set_page_config(page_title="MLB Strategy Engine", layout="wide")

# --- 2. DATA ENGINE ---
@st.cache_data
def get_data():
    try:
        # Load raw files
        sc = pd.read_csv("stat_cast_two.csv")
        war = pd.read_csv("war_stats.csv")
        con = pd.read_csv("spotrac.csv")

        # Helper: Normalize Names
        import unicodedata

        def normalize_name(name):
            if pd.isna(name): return np.nan
            
            # 1. Strip Accents (Turns Peña -> Pena)
            name = str(name)
            name = "".join(c for c in unicodedata.normalize('NFD', name)
                          if unicodedata.category(c) != 'Mn')
            
            # 2. Convert to lowercase and strip whitespace
            s = name.lower().strip()
            
            # 3. Handle "Last, First" vs "First Last"
            if "," in s:
                parts = [p.strip() for p in s.split(",", 1)]
                s = f"{parts[1]} {parts[0]}"
                
            # 4. Remove ALL non-alphabet characters (strips hyphens, periods, etc.)
            # This turns "crow-armstrong" into "crow armstrong"
            s = re.sub(r"[^a-z\s]", " ", s)
            
            # 5. Remove suffixes (jr, sr, iii) and collapse extra spaces
            s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
            s = " ".join(s.split()) 
            
            return s

        # Helper: Clean Currency
        def clean_currency(value):
            if isinstance(value, str):
                clean_val = value.replace('$', '').replace(',', '').strip()
                try: return float(clean_val)
                except: return 0.0
            return value

        # Process Keys
        sc['player_key'] = sc[sc.columns[0]].apply(normalize_name)
        war['player_key'] = war['Player'].apply(normalize_name)
        con['player_key'] = con['Player'].apply(normalize_name)
        
        # Clean numeric columns
        con['AAV'] = con['AAV'].apply(clean_currency)
        con['Start'] = pd.to_numeric(con['Start'], errors='coerce').fillna(2024).astype(int)
        con['End'] = pd.to_numeric(con['End'], errors='coerce').fillna(2025).astype(int)
        # Filter for 2025 season contracts (Start <= 2025 and End >= 2025)
        con = con[(con['Start'] <= 2025) & (con['End'] >= 2025)]
        war['WAR'] = pd.to_numeric(war['WAR'], errors='coerce').fillna(0)

        # Handle players with multiple teams - prefer "2TM" or highest WAR
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
    
        # --- MANUAL DATA OVERRIDES ---
        # Jeremy Pena (2026 Arb Salary): $9,475,000
        # Pete Crow-Armstrong (2026 Est): $780,000 (unless extension signed)
        
        overrides = {
            "jeremy pena": 9475000,
            "pete crow armstrong": 780000,
            "christian walker": 22000000 # Estimated new AAV
        }
        
        for player, salary in overrides.items():
            df_result.loc[df_result['player_key'] == player, 'AAV'] = salary
        
        # Fill missing values for math
        df_result['WAR'] = df_result['WAR'].fillna(0)
        df_result['AAV'] = df_result['AAV'].fillna(0)
        df_result['Team'] = df_result['Team'].fillna("Free Agent")
        df_result['player_key'] = df_result['player_key'].astype(str)

        # STRATEGIC MATH
        MARKET_RATE = 8500000 
        df_result['Market_Value'] = df_result['WAR'] * MARKET_RATE
        df_result['Surplus'] = df_result['Market_Value'] - df_result['AAV']
        df_result['Density_Score'] = df_result['WAR'].astype(float)
        df_result['Status'] = np.where(df_result['AAV'] < 1000000, "Pre-Arb", "Veteran/FA")
        
        def get_reason(row):
            if row['Surplus'] > 5e6: return "✅ Core Asset: High Surplus"
            if row['Surplus'] < -5e6: return "❌ Toxic: Major Underperformance"
            return "🔄 Fair Value"
        
        df_result['Strategic_Note'] = df_result.apply(get_reason, axis=1)

        return df_result
        
    except Exception as e:
        st.error(f"Error inside get_data: {e}")
        return None

# --- 3. INITIALIZE APP ---
df = get_data()

if df is not None:
    # Sidebar
    st.sidebar.title("💎 Strategy Room")
    nav = st.sidebar.radio("Navigation", ["Central Dashboard", "Team Strategy Room", "Scavenger List"])
    
    # Global Filter
    filtered_df = df[df['AAV'] > 0]

    # --- VIEW: CENTRAL DASHBOARD ---
    if nav == "Central Dashboard":
        st.title("⚾ 2025 MLB Market Intelligence")
        m1, m2, m3 = st.columns(3)
        m1.metric("Market $/WAR", "$8.5M")
        m2.metric("League Surplus", f"${(filtered_df['Surplus'].sum()/1e6):.1f}M")
        m3.metric("Top Win Density", f"{filtered_df['Density_Score'].max():.1f}")

        st.divider()
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.success("### 🚀 Top Surplus Values")
            top_surplus = filtered_df[(~filtered_df['Pos'].isin(['SP', 'RP']))].sort_values('Surplus', ascending=False).head(10)[['player_key', 'WAR', 'AAV', 'Surplus']].copy()
            for col in top_surplus.columns:
                if col == 'player_key':
                    top_surplus[col] = top_surplus[col].astype(str)
                elif col in ['AAV', 'Surplus']:
                    top_surplus[col] = top_surplus[col].apply(lambda x: f"${int(x):,}")
                else:
                    top_surplus[col] = top_surplus[col].apply(lambda x: f"{x:.1f}")
            top_surplus_display = top_surplus.reset_index(drop=True)
            top_surplus_display.index = top_surplus_display.index + 1
            st.write(top_surplus_display.to_html(escape=False), unsafe_allow_html=True)
        with c2:
            st.error("### 📉 Top Efficiency Leaks")
            top_leaks = filtered_df[(~filtered_df['Pos'].isin(['SP', 'RP']))].sort_values('Surplus', ascending=True).head(10)[['player_key', 'WAR', 'AAV', 'Surplus']].copy()
            for col in top_leaks.columns:
                if col == 'player_key':
                    top_leaks[col] = top_leaks[col].astype(str)
                elif col in ['AAV', 'Surplus']:
                    top_leaks[col] = top_leaks[col].apply(lambda x: f"${int(x):,}")
                else:
                    top_leaks[col] = top_leaks[col].apply(lambda x: f"{x:.1f}")
            top_leaks_display = top_leaks.reset_index(drop=True)
            top_leaks_display.index = top_leaks_display.index + 1
            st.write(top_leaks_display.to_html(escape=False), unsafe_allow_html=True)
        with c3:
            st.info("### 🏆 Team Surplus Ranking")
            team_surplus = filtered_df[(~filtered_df['Pos'].isin(['SP', 'RP'])) & (~filtered_df['Team'].isin(['2TM', '3TM']))].groupby('Team')['Surplus'].sum().sort_values(ascending=False).reset_index()
            team_surplus.columns = ['Team', 'Total Surplus']
            team_surplus['Total Surplus'] = team_surplus['Total Surplus'].apply(lambda x: f"${int(x):,}")
            team_surplus_display = team_surplus.reset_index(drop=True)
            team_surplus_display.index = team_surplus_display.index + 1
            st.write(team_surplus_display.to_html(escape=False), unsafe_allow_html=True)

    # --- VIEW: TEAM STRATEGY ROOM ---
    elif nav == "Team Strategy Room":
        # Filter out "2TM" and "3TM" from team options
        team_options = sorted([team for team in df['Team'].unique() if team not in ['2TM', '3TM']])
        selected_team = st.selectbox("Select Organization", team_options)
        team_df = filtered_df[filtered_df['Team'] == selected_team].copy()
        
        # Replace headers with total surplus metric
        total_surplus = filtered_df['Surplus'].sum()
        st.metric("Total League Surplus", f"${total_surplus:,.0f}")
        
        st.divider()
        st.subheader("Complete Roster Audit")
        
        team_roster_data = df[
            (df['Team'] == selected_team) & 
            (~df['Pos'].isin(['SP', 'RP'])) &  # Exclude pitchers
            (df['AAV'] > 0)  # Exclude zero AAV players
        ][['player_key', 'WAR', 'AAV', 'Surplus', 'Strategic_Note']].sort_values('Surplus').copy()
        
        for col in team_roster_data.columns:
            if col == 'player_key':
                team_roster_data[col] = team_roster_data[col].astype(str)
            elif col in ['AAV', 'Surplus']:
                team_roster_data[col] = team_roster_data[col].apply(lambda x: f"${int(x):,}")
            elif col == 'WAR':
                team_roster_data[col] = team_roster_data[col].apply(lambda x: f"{x:.1f}")
            else:
                team_roster_data[col] = team_roster_data[col].astype(str)
        
        team_roster_display = team_roster_data.reset_index(drop=True)
        team_roster_display.index = team_roster_display.index + 1
        st.write(team_roster_display.to_html(escape=False), unsafe_allow_html=True)

    # --- VIEW: SCAVENGER LIST ---
    elif nav == "Scavenger List":
        st.title("🕵️ Scavenger: High-Skill / Low-Cost Targets")
        st.info("Searching for assets with High WAR but Low AAV across the league.")
        
        scavenger = df[(~df['Pos'].isin(['SP', 'RP'])) & (df['AAV'] < 2000000) & (df['AAV'] > 0) & (df['WAR'] > 1.2)].sort_values('Surplus', ascending=False)
        scavenger_data = scavenger[['player_key', 'Team', 'WAR', 'AAV', 'Surplus']].copy()
        
        for col in scavenger_data.columns:
            if col == 'player_key':
                scavenger_data[col] = scavenger_data[col].astype(str)
            elif col == 'Team':
                scavenger_data[col] = scavenger_data[col].astype(str)
            elif col in ['AAV', 'Surplus']:
                scavenger_data[col] = scavenger_data[col].apply(lambda x: f"${int(x):,}")
            elif col == 'WAR':
                scavenger_data[col] = scavenger_data[col].apply(lambda x: f"{x:.1f}")
        
        scavenger_display = scavenger_data.reset_index(drop=True)
        scavenger_display.index = scavenger_display.index + 1
        st.write(scavenger_display.to_html(escape=False), unsafe_allow_html=True)

else:
    st.error("Could not load data. Please check your CSV files and column names.")