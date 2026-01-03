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
    # Season input primarily for your future database logic
    current_season = st.text_input("Season", value="2025-26")
    
    st.divider()
    st.write("🗓️ **Schedule Settings**")
    # Defaults to Today. You can change this to see other days.
    target_date = st.date_input("Target Date", value=pd.to_datetime("today"))
    
    st.divider()
    st.info("ℹ️ **Note:** This app now uses ESPN's API to bypass Cloud IP blocks.")

# --- 1. ROBUST DATA FETCHING (ESPN VERSION) ---
@st.cache_data(ttl=3600)
def fetch_nba_schedule(date_str):
    """
    Fetches the schedule from ESPN's public API.
    Much more reliable for Streamlit Cloud than NBA.com.
    """
    # ESPN expects date format YYYYMMDD (e.g., 20260103)
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
        {"Away Team": "BOS", "Home Team": "NYK", "Vegas Line (Home)": -2.5},
        {"Away Team": "LAL", "Home Team": "GSW", "Vegas Line (Home)": 3.0}
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

# --- 4. PROJECTIONS (Placeholder Logic) ---
# NOTE: In your final version, you will replace this 'dummy_ratings' dictionary 
# with the actual 'team_ratings' dictionary from your Ridge Regression model.

st.subheader("2. Live Projections")

# Placeholder Ratings (Just to show the math works)
# 'HFA' is usually around -2.7 points (Home Advantage)
hfa = -2.7
dummy_ratings = {
    'BOS': -8.0, 'NYK': -2.0, 'PHI': -3.5, 'MIL': -4.0, 'CLE': -3.0,
    'DEN': -5.0, 'MIN': -4.5, 'OKC': -6.0, 'LAC': -1.0, 'PHX': -2.5,
    'LAL': 0.5, 'GSW': 1.0, 'MIA': -1.5, 'ORL': -1.0, 'IND': 0.0
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
    # Example: BOS(-8) vs NYK(-2) at BOS
    # (-8 - -2) + -2.7 = -6 + -2.7 = -8.7 (BOS favored by 8.7)
    model_line = (r_h - r_a) + hfa
    
    # EDGE: Model - Vegas
    # If Model = -8.7, Vegas = -5.5
    # Edge = -3.2 (Model thinks Home is STRONGER favorite) -> BET HOME
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
results_df = pd.DataFrame(results)

def color_signal(val):
    color = ''
    if 'BET' in str(val):
        color = 'background-color: #d4edda; color: #155724; font-weight: bold'
    return color

st.dataframe(
    results_df.style.map(color_signal, subset=['Signal']), 
    use_container_width=True,
    hide_index=True
)
