import streamlit as st
import pandas as pd
import numpy as np
import requests
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA Market Ratings", layout="wide")
st.title("🏀 NBA Market-Implied Ratings")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    current_season = st.text_input("Season", value="2025-26")
    
    st.divider()
    st.write("🗓️ **Schedule Settings**")
    # Defaults to Today. You can change this to see other days.
    target_date = st.date_input("Target Date", value=pd.to_datetime("today"))
    
    st.divider()
    st.info("ℹ️ **Note:** This app uses ESPN's API to bypass Cloud IP blocks.")

# --- 1. ROBUST DATA FETCHING (ESPN VERSION) ---
@st.cache_data(ttl=3600)
def fetch_nba_schedule(date_str):
    """
    Fetches the schedule from ESPN's public API.
    Much more reliable for Streamlit Cloud than NBA.com.
    """
    # ESPN expects date format YYYYMMDD
    espn_date = date_str.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={espn_date}"
    
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        
        games = []
        if 'events' not in data:
            return pd.DataFrame()
            
        for event in data['events']:
            competition = event['competitions'][0]
            competitors = competition['competitors']
            
            # ESPN lists competitors as [Home, Away] or vice versa. 
            # We must check the 'homeAway' tag.
            home_team = next(filter(lambda x: x['homeAway'] == 'home', competitors))
            away_team = next(filter(lambda x: x['homeAway'] == 'away', competitors))
            
            # Get Abbreviations (e.g., 'LAL', 'BOS')
            h_abbr = home_team['team']['abbreviation']
            a_abbr = away_team['team']['abbreviation']
            
            games.append({
                "HOME_TEAM_ID": h_abbr, 
                "VISITOR_TEAM_ID": a_abbr
            })
            
        return pd.DataFrame(games)

    except Exception as e:
        st.error(f"❌ Error fetching from ESPN: {e}")
        return pd.DataFrame()

# --- 2. MAIN LOGIC ---

# Status Indicator
status_text = st.empty()
status_text.info(f"⏳ Fetching Schedule for {target_date}...")

# Fetch Schedule
date_str = target_date.strftime('%Y-%m-%d')
schedule_df = fetch_nba_schedule(date_str)

if schedule_df.empty:
    status_text.warning("⚠️ No games found for this date (or API issue). Using dummy data.")
    # Fallback Data so the app doesn't look empty
    input_rows = [
        {"Away Team": "BOS", "Home Team": "NY", "Vegas Line (Home)": -2.5},
        {"Away Team": "LAL", "Home Team": "GS", "Vegas Line (Home)": 3.0}
    ]
else:
    status_text.success(f"✅ Loaded {len(schedule_df)} Games from ESPN!")
    
    # Build Input Rows from Real Schedule
    input_rows = []
    for _, row in schedule_df.iterrows():
        input_rows.append({
            "Away Team": row['VISITOR_TEAM_ID'],
            "Home Team": row['HOME_TEAM_ID'],
            "Vegas Line (Home)": 0.0 # Default line
        })

# --- 3. DASHBOARD ---
st.subheader("1. Matchup Settings")
input_df = pd.DataFrame(input_rows)

# Editable Table
edited_df = st.data_editor(
    input_df, 
    num_rows="dynamic",
    column_config={
        "Vegas Line (Home)": st.column_config.NumberColumn(
            "Vegas Line (Home)",
            help="Enter -5.5 if Home is favored by 5.5",
            format="%.1f"
        )
    }
)

# --- 4. PROJECTIONS ---
st.subheader("2. Live Projections")

# ⚠️ PLACEHOLDER RATINGS ⚠️
# In the future, you can replace this dictionary with one loaded from a CSV 
# or calculated from 'Net Rating'.
# These keys MUST match ESPN abbreviations (e.g. 'UTAH', 'NY', 'GS', 'NO', 'SA').
hfa = -2.7 
dummy_ratings = {
    'BOS': -9.0, 'OKC': -7.5, 'MIN': -6.0, 'DEN': -5.5, 'NY': -4.5,
    'PHI': -4.0, 'LAC': -3.5, 'CLE': -3.0, 'MIL': -3.0, 'DAL': -2.5,
    'PHX': -2.5, 'NO': -2.0, 'IND': -1.5, 'MIA': -1.0, 'ORL': -1.0,
    'SAC': -0.5, 'GS': 0.0, 'LAL': 0.0, 'HOU': 1.0, 'CHI': 2.0,
    'ATL': 2.5, 'BKN': 3.0, 'UTAH': 3.5, 'TOR': 4.0, 'MEM': 4.5,
    'CHA': 5.0, 'POR': 5.5, 'SA': 6.0, 'WAS': 7.0, 'DET': 8.0
}

results = []

for _, row in edited_df.iterrows():
    h = row['Home Team']
    a = row['Away Team']
    vegas = row['Vegas Line (Home)']
    
    # Lookup Ratings (Default to 0.0 if not found)
    r_h = dummy_ratings.get(h, 0.0)
    r_a = dummy_ratings.get(a, 0.0)
    
    # MODEL FORMULA: (Home Rating - Away Rating) + HFA
    # Example: BOS(-9) vs NY(-4.5)
    # (-9 - -4.5) + -2.7 = -7.2 (BOS favored by 7.2)
    model_line = (r_h - r_a) + hfa
    
    # EDGE: Model - Vegas
    edge = model_line - vegas
    
    # SIGNAL LOGIC
    signal = "PASS"
    if edge < -2.0: 
        signal = f"BET {h}"
    elif edge > 2.0: 
        signal = f"BET {a}"
    
    results.append({
        "Matchup": f"{a} @ {h}",
        "Model Line": round(model_line, 1),
        "Vegas Line": vegas,
        "Edge": round(edge, 1),
        "Signal": signal
    })

# Display Results
if results:
    results_df = pd.DataFrame(results)

    # Styling Helper
    def color_signal(val):
        color = ''
        if 'BET' in str(val):
            color = 'background-color: #d4edda; color: #155724; font-weight: bold'
        return color

    # FIX: Use 'applymap' instead of 'map' for compatibility
    st.dataframe(
        results_df.style.applymap(color_signal, subset=['Signal']), 
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Waiting for matchups...")
