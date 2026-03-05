import streamlit as st
import pandas as pd
import numpy as np
import re

# --- AI SCOUT REPORTS ---
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False

def generate_scout_report(player_data):
    """Generate AI-powered scout report for a player"""
    if not AI_AVAILABLE:
        return "AI scout reports unavailable - please configure OpenAI API key in secrets."

    prompt = f"""
    Analyze this MLB player and explain why they have a low WAR despite their swing metrics:

    Player: {player_data['name']} ({player_data['position']})
    WAR: {player_data['war']}
    Meatball Swing %: {player_data['meatball_pct']}%
    Poorly Under %: {player_data.get('under_pct', 'N/A')}%
    Poorly Topped %: {player_data.get('topped_pct', 'N/A')}%
    Avg Swing Length: {player_data['swing_len']} ft
    Zone Swinging Rate: {player_data.get('zone_swing_rate', 'N/A')}

    Focus on mechanical 'Process Leaks' like swing path consistency, launch angle issues,
    or contact quality problems that might explain the disconnect between their approach
    and actual performance. Be specific and actionable.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an elite MLB scout with 30+ years experience analyzing hitters. Provide detailed, technical analysis focusing on mechanical issues and process leaks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

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

            name = str(name)
            name = "".join(c for c in unicodedata.normalize('NFD', name)
                          if unicodedata.category(c) != 'Mn')

            s = name.lower().strip()

            if "," in s:
                parts = [p.strip() for p in s.split(",", 1)]
                s = f"{parts[1]} {parts[0]}"

            s = re.sub(r"[^a-z\s]", " ", s)

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

        # THE MERGE
        df_result = war[['player_key', 'WAR', 'Team', 'Pos']].merge(sc_agg, on="player_key", how="left")
        df_result = df_result.merge(con[['player_key', 'AAV', 'Pos']], on="player_key", how="left", suffixes=('', '_contract'))

        # Use contract position if available, otherwise use WAR position
        df_result['Pos'] = df_result['Pos_contract'].fillna(df_result['Pos'])
        df_result = df_result.drop('Pos_contract', axis=1)

        # --- MANUAL DATA OVERRIDES ---
        overrides = {
            "jeremy pena": 9475000,
            "pete crow armstrong": 780000,
            "christian walker": 22000000
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
    nav = st.sidebar.radio("Navigation", ["Central Dashboard", "Team Strategy Room", "Breakout Stars"])

    # Global Filter
    filtered_df = df[df['AAV'] > 0]

    # --- SHARED: BREAKOUT SCORE (used in Team Strategy Room + Scavenger List) ---
    PITCHER_POS = ['SP', 'RP', 'P', '1', '/1']
    BREAKOUT_METRICS = [
        'meatball_swing_percent',   # high = good
        'oz_swing_percent',         # high = BAD (inverted)
        'barrel_batted_rate',       # high = good
        'sweet_spot_percent',       # high = good
        'exit_velocity_avg',        # high = good
        'fast_swing_rate',          # high = good
        'attack_angle',             # high = good
        'launch_angle_avg',         # high = good
        'iz_contact_percent',       # high = good
    ]
    all_pos = df[~df['Pos'].isin(PITCHER_POS)].dropna(subset=BREAKOUT_METRICS).copy()
    for _col in BREAKOUT_METRICS:
        if _col == 'oz_swing_percent':
            all_pos[f'_pct_{_col}'] = all_pos[_col].rank(pct=True, ascending=False)
        else:
            all_pos[f'_pct_{_col}'] = all_pos[_col].rank(pct=True, ascending=True)
    _pct_cols = [f'_pct_{m}' for m in BREAKOUT_METRICS]
    all_pos['Breakout Score'] = (all_pos[_pct_cols].mean(axis=1) * 100).round(1)
    breakout_pool = all_pos[
        ((all_pos['WAR'] < 3) & (all_pos['player_age'] < 28) & (all_pos['Breakout Score'] >= 40)) |
        ((all_pos['player_age'] <= 25) & (all_pos['WAR'] > 4.5))
    ].copy()

    # --- VIEW: CENTRAL DASHBOARD ---
    if nav == "Central Dashboard":
        st.title("⚾ 2025 MLB Market Intelligence")
        st.caption("**Surplus = (WAR × $8.5M) − Contract AAV** — 1 WAR ≈ $8.5M on the open market. Positive surplus = team-friendly deal; negative surplus = overpaid relative to production.")
        st.divider()
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown("##### 🚀 Top Surplus Values")
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
            st.markdown("##### 📉 Top Efficiency Leaks")
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
            st.markdown("##### 🏆 Team Surplus Ranking")
            team_surplus = filtered_df[(~filtered_df['Pos'].isin(['SP', 'RP'])) & (~filtered_df['Team'].isin(['2TM', '3TM']))].groupby('Team')['Surplus'].sum().sort_values(ascending=False).reset_index()
            team_surplus.columns = ['Team', 'Total Surplus']
            team_surplus['Total Surplus'] = team_surplus['Total Surplus'].apply(lambda x: f"${int(x):,}")
            team_surplus_display = team_surplus.reset_index(drop=True)
            team_surplus_display.index = team_surplus_display.index + 1
            st.write(team_surplus_display.to_html(escape=False), unsafe_allow_html=True)

    # --- VIEW: TEAM STRATEGY ROOM ---
    elif nav == "Team Strategy Room":
        ALL_MLB_TEAMS = sorted([
            'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
            'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
            'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
        ])

        st.title("⚾ Team Strategy Room")
        selected_team = st.selectbox("Select Organization", ALL_MLB_TEAMS, label_visibility="collapsed")

        st.markdown("---")

        # --- ROSTER SECTION ---
        team_raw = df[
            (df['Team'] == selected_team) &
            (~df['Pos'].isin(PITCHER_POS)) &
            (df['AAV'] > 0)
        ][['player_key', 'WAR', 'AAV', 'Surplus', 'Strategic_Note']].sort_values('WAR', ascending=False).copy()

        roster_count = len(team_raw)
        core_count = (team_raw['Strategic_Note'].str.contains('Core Asset', na=False)).sum()
        toxic_count = (team_raw['Strategic_Note'].str.contains('Toxic', na=False)).sum()
        fair_count = (team_raw['Strategic_Note'].str.contains('Fair Value', na=False)).sum()
        team_breakout = breakout_pool[breakout_pool['Team'] == selected_team].copy()
        breakout_count = len(team_breakout)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Players", roster_count)
        m2.metric("✅ Core Assets", core_count)
        m3.metric("❌ Toxic Assets", toxic_count)
        m4.metric("🔄 Fair Value", fair_count)
        m5.metric("🚀 Breakout Candidates", breakout_count)

        st.markdown("---")

        col_roster, col_breakout = st.columns([1, 1])

        with col_roster:
            st.markdown("#### Position Player Roster")
            team_roster_data = team_raw.rename(columns={'player_key': 'Player', 'Strategic_Note': 'Note'})
            team_roster_data['WAR'] = team_roster_data['WAR'].apply(lambda x: f"{x:.1f}")
            team_roster_data['AAV'] = team_roster_data['AAV'].apply(lambda x: f"${int(x):,}")
            team_roster_data['Surplus'] = team_roster_data['Surplus'].apply(lambda x: f"${int(x):,}")
            team_roster_data['Note'] = team_roster_data['Note'].astype(str)
            team_roster_display = team_roster_data.reset_index(drop=True)
            team_roster_display.index = team_roster_display.index + 1
            st.write(
                "<div style='margin-top: 8px;'>" +
                team_roster_display.to_html(escape=False) +
                "</div>",
                unsafe_allow_html=True
            )

        with col_breakout:
            st.markdown("#### 🚀 Breakout Candidates")
            st.caption("Under 28 + WAR < 3 + Score ≥ 40, or age ≤ 25 + WAR > 4.5 — percentile-ranked against all leaguewide position players")
            if team_breakout.empty:
                st.info("No breakout candidates found for this team with sufficient Statcast data.")
            else:
                display_cols = {
                    'player_key': 'Player',
                    'player_age': 'Age',
                    'WAR': 'WAR',
                    'meatball_swing_percent': 'Meatball%',
                    'oz_swing_percent': 'Chase%',
                    'barrel_batted_rate': 'Barrel%',
                    'sweet_spot_percent': 'Sweet Spot%',
                    'exit_velocity_avg': 'Exit Velo',
                    'fast_swing_rate': 'Fast Swing%',
                    'attack_angle': 'Attack Angle',
                    'launch_angle_avg': 'Launch Angle',
                    'iz_contact_percent': 'IZ Contact%',
                    'Breakout Score': 'Breakout Score',
                }
                bc = team_breakout.sort_values('Breakout Score', ascending=False)[list(display_cols.keys())].rename(columns=display_cols).copy()
                bc['Age'] = bc['Age'].apply(lambda x: f"{int(x)}")
                bc['WAR'] = bc['WAR'].apply(lambda x: f"{x:.1f}")
                bc['Meatball%'] = bc['Meatball%'].apply(lambda x: f"{x:.1f}%")
                bc['Chase%'] = bc['Chase%'].apply(lambda x: f"{x:.1f}%")
                bc['Barrel%'] = bc['Barrel%'].apply(lambda x: f"{x:.1f}%")
                bc['Sweet Spot%'] = bc['Sweet Spot%'].apply(lambda x: f"{x:.1f}%")
                bc['Exit Velo'] = bc['Exit Velo'].apply(lambda x: f"{x:.1f}")
                bc['Fast Swing%'] = bc['Fast Swing%'].apply(lambda x: f"{x:.1f}%")
                bc['Attack Angle'] = bc['Attack Angle'].apply(lambda x: f"{x:.1f}°")
                bc['Launch Angle'] = bc['Launch Angle'].apply(lambda x: f"{x:.1f}°")
                bc['IZ Contact%'] = bc['IZ Contact%'].apply(lambda x: f"{x:.1f}%")
                bc['Breakout Score'] = bc['Breakout Score'].apply(lambda x: f"{x:.1f}")
                bc = bc.reset_index(drop=True)
                bc.index = bc.index + 1
                st.write(
                    "<div style='margin-top: 8px;'>" +
                    bc.to_html(escape=False) +
                    "</div>",
                    unsafe_allow_html=True
                )

    # --- VIEW: BREAKOUT STARS ---
    elif nav == "Breakout Stars":
        st.title("🚀 Leaguewide Breakout Stars")
        st.caption("Under 28 + WAR < 3 + Score ≥ 40, or age ≤ 25 + WAR > 4.5 — ranked by Breakout Score against all leaguewide position players.")

        # Compute 60th percentile benchmarks across ALL position players
        pct60_metrics = {
            'meatball_swing_percent': 'Meatball%',
            'oz_swing_percent': 'Chase%',
            'barrel_batted_rate': 'Barrel%',
            'sweet_spot_percent': 'Sweet Spot%',
            'exit_velocity_avg': 'Exit Velo',
            'fast_swing_rate': 'Fast Swing%',
            'attack_angle': 'Attack Angle',
            'launch_angle_avg': 'Launch Angle',
            'iz_contact_percent': 'IZ Contact%',
        }
        benchmark_row = {
            'Player': '📊 60th Pct (All Pos)',
            'Team': '—',
            'Age': '—',
            'WAR': f"{all_pos['WAR'].quantile(0.6):.1f}",
        }
        for raw_col, display_col in pct60_metrics.items():
            val = all_pos[raw_col].quantile(0.6)
            if display_col in ['Attack Angle', 'Launch Angle']:
                benchmark_row[display_col] = f"{val:.1f}°"
            elif display_col == 'Exit Velo':
                benchmark_row[display_col] = f"{val:.1f}"
            else:
                benchmark_row[display_col] = f"{val:.1f}%"
        benchmark_row['Breakout Score'] = '—'

        scavenger_display_cols = {
            'player_key': 'Player',
            'Team': 'Team',
            'player_age': 'Age',
            'WAR': 'WAR',
            'meatball_swing_percent': 'Meatball%',
            'oz_swing_percent': 'Chase%',
            'barrel_batted_rate': 'Barrel%',
            'sweet_spot_percent': 'Sweet Spot%',
            'exit_velocity_avg': 'Exit Velo',
            'fast_swing_rate': 'Fast Swing%',
            'attack_angle': 'Attack Angle',
            'launch_angle_avg': 'Launch Angle',
            'iz_contact_percent': 'IZ Contact%',
            'Breakout Score': 'Breakout Score',
        }
        sc_data = breakout_pool.sort_values('Breakout Score', ascending=False)[list(scavenger_display_cols.keys())].rename(columns=scavenger_display_cols).copy()
        sc_data['Age'] = sc_data['Age'].apply(lambda x: f"{int(x)}")
        sc_data['WAR'] = sc_data['WAR'].apply(lambda x: f"{x:.1f}")
        sc_data['Meatball%'] = sc_data['Meatball%'].apply(lambda x: f"{x:.1f}%")
        sc_data['Chase%'] = sc_data['Chase%'].apply(lambda x: f"{x:.1f}%")
        sc_data['Barrel%'] = sc_data['Barrel%'].apply(lambda x: f"{x:.1f}%")
        sc_data['Sweet Spot%'] = sc_data['Sweet Spot%'].apply(lambda x: f"{x:.1f}%")
        sc_data['Exit Velo'] = sc_data['Exit Velo'].apply(lambda x: f"{x:.1f}")
        sc_data['Fast Swing%'] = sc_data['Fast Swing%'].apply(lambda x: f"{x:.1f}%")
        sc_data['Attack Angle'] = sc_data['Attack Angle'].apply(lambda x: f"{x:.1f}°")
        sc_data['Launch Angle'] = sc_data['Launch Angle'].apply(lambda x: f"{x:.1f}°")
        sc_data['IZ Contact%'] = sc_data['IZ Contact%'].apply(lambda x: f"{x:.1f}%")
        sc_data['Breakout Score'] = sc_data['Breakout Score'].apply(lambda x: f"{x:.1f}")
        sc_data = sc_data.reset_index(drop=True)

        # Prepend benchmark row
        benchmark_df = pd.DataFrame([benchmark_row], index=['—'])
        sc_data.index = sc_data.index + 1
        sc_data = pd.concat([benchmark_df, sc_data])

        st.write(
            "<div style='margin-top: 8px;'>" +
            sc_data.to_html(escape=False) +
            "</div>",
            unsafe_allow_html=True
        )

else:
    st.error("Could not load data. Please check your CSV files and column names.")
