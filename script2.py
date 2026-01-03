import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import requests
import io
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA Market Ratings", layout="wide")
st.title("🏀 NBA Market-Implied Ratings")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    current_season = st.text_input("Season (e.g., 2024-25)", value="2025-26")
    decay_alpha = st.slider("Recency Decay (Alpha)", 0.01, 0.20, 0.10)
    
    st.divider()
    st.write("🗓️ **Schedule Settings**")
    target_date = st.date_input("Target Date", value=pd.to_datetime("today"))
    
    # Debug Mode
    show_debug = st.checkbox("Show Debug Info", value=True)

# --- 1. ROBUST DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_nba_schedule(date_str):
    """
    Fetches the scoreboard for a specific date using direct requests 
    to bypass Cloud IP blocks.
    """
    # Endpoint for the specific day's scoreboard
    url = "https://stats.nba.com/stats/scoreboardv2"
    
    params = {
        'GameDate': date_str,
        'LeagueID': '00',
        'DayOffset': '0'
    }
    
    # HEADERS ARE CRITICAL: This pretends to be a Chrome Browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'Accept-Language': 'en-US,en;q=0.9',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true'
    }
    
    try:
        # Timeout is 5 seconds. If NBA.com ignores us, we fail fast.
        r = requests.get(url, params=params, headers=headers, timeout=5)
        r.raise_for_status()
        
        data = r.json()
        headers_list = data['resultSets'][0]['headers']
        row_set = data['resultSets'][0]['rowSet']
        
        df = pd.DataFrame(row_set, columns=headers_list)
        return df
        
    except requests.exceptions.Timeout:
        st.error("❌ Connection Timed Out. NBA.com blocked the request.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error fetching schedule: {e}")
        return pd.DataFrame()

# --- 2. MAIN LOGIC ---

# A. STATUS UPDATE
status_text = st.empty()
status_text.info("⏳ Fetching Schedule from NBA.com...")

# B. FETCH SCHEDULE
date_str = target_date.strftime('%Y-%m-%d')
if show_debug:
    st.write(f"Attempting to fetch data for: {date_str}")

schedule_df = fetch_nba_schedule(date_str)

if schedule_df.empty:
    status_text.error("⚠️ Could not load schedule. (Cloud IP likely blocked by NBA)")
    st.warning("Since the auto-schedule failed, please manually select teams below.")
    # Create an empty dataframe so the app doesn't crash
    current_matchups = pd.DataFrame(columns=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
else:
    status_text.success("✅ Schedule Loaded!")
    if show_debug:
        st.dataframe(schedule_df[['GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']].head())
    current_matchups = schedule_df

# --- 3. TEAM MAPPING (Static Fallback) ---
# Since we might not be able to fetch the full team list, we hardcode a few IDs for safety
# In a full app, you'd fetch this once and cache it.
TEAM_MAP = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
    1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
    1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
    1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
    1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
    1610612762: 'UTA', 1610612764: 'WAS'
}

# --- 4. BUILD THE DASHBOARD ---
st.subheader("Matchup Settings")

# Convert Schedule to Readable Input Table
input_rows = []

if not schedule_df.empty:
    for _, row in schedule_df.iterrows():
        h_id = row['HOME_TEAM_ID']
        a_id = row['VISITOR_TEAM_ID']
        h_abbr = TEAM_MAP.get(h_id, str(h_id))
        a_abbr = TEAM_MAP.get(a_id, str(a_id))
        
        input_rows.append({
            "Away Team": a_abbr,
            "Home Team": h_abbr,
            "Vegas Line (Home)": -5.5 # Placeholder
        })
else:
    # Default rows if schedule fails
    input_rows = [
        {"Away Team": "BOS", "Home Team": "NYK", "Vegas Line (Home)": -2.5},
        {"Away Team": "LAL", "Home Team": "GSW", "Vegas Line (Home)": 3.0}
    ]

input_df = pd.DataFrame(input_rows)

# EDITABLE TABLE
edited_df = st.data_editor(
    input_df, 
    num_rows="dynamic",
    column_config={
        "Vegas Line (Home)": st.column_config.NumberColumn(format="%.1f")
    }
)

# --- 5. DUMMY RATINGS (To prevent crash) ---
# In this debug version, I'm using dummy ratings. 
# Once the schedule works, we will re-connect the Ridge Regression logic.
ratings = {k: np.random.uniform(-10, 10) for k in TEAM_MAP.values()}
hfa = -2.5

st.subheader("Live Projections (Debug Mode)")

results = []
for _, row in edited_df.iterrows():
    h = row['Home Team']
    a = row['Away Team']
    vegas = row['Vegas Line (Home)']
    
    # Safe Get
    r_h = ratings.get(h, 0.0)
    r_a = ratings.get(a, 0.0)
    
    # Calc
    model_line = (r_h - r_a) + hfa
    edge = model_line - vegas
    
    signal = "PASS"
    if edge < -2.5: signal = f"BET {h}"
    elif edge > 2.5: signal = f"BET {a}"
    
    results.append({
        "Matchup": f"{a} @ {h}",
        "Model": round(model_line, 1),
        "Vegas": vegas,
        "Edge": round(edge, 1),
        "Signal": signal
    })

st.dataframe(pd.DataFrame(results).style.applymap(
    lambda x: 'background-color: #d4edda' if "BET" in str(x) else '', subset=['Signal']
))
