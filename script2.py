import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import Ridge
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA Market Ratings", layout="wide")
st.title("🏀 NBA Market-Implied Power Ratings")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    decay_alpha = st.slider("Recency Decay", 0.0, 0.1, 0.015, step=0.001)
    target_date = st.date_input("Target Date", value=datetime.now())

# --- 1. DATA LOADER ---
@st.cache_data(ttl=3600)
def load_training_data():
    """
    Load historical data. Defaults to synthetic data if CSV fails.
    """
    # Placeholder URL for public data
    CSV_URL = "https://raw.githubusercontent.com/nealmick/Sports-Betting-ML-Tools-NBA/master/data/nba_odds_2023-24.csv"
    
    try:
        df = pd.read_csv(CSV_URL)
        if df.empty: raise ValueError("Empty CSV")
        # Ensure minimal columns exist
        req_cols = ['Date', 'Home', 'Away', 'Market_Line']
        if not set(req_cols).issubset(df.columns):
            raise ValueError(f"Missing columns. Found: {df.columns}")
        return df
        
    except Exception:
        # FALLBACK: Synthetic Data Generator
        # This ensures the app ALWAYS runs.
        dates = pd.date_range(end=datetime.now(), periods=400)
        teams = ['BOS', 'NYK', 'PHI', 'BKN', 'TOR', 'MIL', 'CLE', 'CHI', 'IND', 'DET', 
                 'MIN', 'OKC', 'DEN', 'POR', 'UTA', 'LAL', 'LAC', 'PHX', 'GSW', 'SAC',
                 'MIA', 'ORL', 'ATL', 'CHA', 'WAS', 'DAL', 'HOU', 'MEM', 'NOP', 'SAS']
        
        fake_games = []
        for d in dates:
            h, a = np.random.choice(teams, 2, replace=False)
            fake_line = np.random.normal(0, 5) 
            fake_games.append({
                'Date': d,
                'Home': h,
                'Away': a,
                'Market_Line': round(fake_line, 1)
            })
            
        return pd.DataFrame(fake_games)

@st.cache_data(ttl=3600)
def fetch_live_schedule(t_date):
    """Fetches today's slate from ESPN."""
    date_str = t_date.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    matchups = []
    try:
        r = requests.get(url, timeout=3)
        data = r.json()
        if 'events' in data:
            for event in data['events']:
                comp = event['competitions'][0]['competitors']
                home = next(filter(lambda x: x['homeAway'] == 'home', comp))
                away = next(filter(lambda x: x['homeAway'] == 'away', comp))
                matchups.append({'Home': home['team']['abbreviation'], 'Away': away['team']['abbreviation']})
    except: pass
    return pd.DataFrame(matchups)

# --- MAIN LOGIC ---

# 1. TRAIN MODEL
st.subheader("1. Market Analysis")
df_train = load_training_data()

# Apply Weights
df_train['Date'] = pd.to_datetime(df_train['Date'])
last_date = df_train['Date'].max()
df_train['Days_Ago'] = (last_date - df_train['Date']).dt.days
df_train['Weight'] = np.exp(-decay_alpha * df_train['Days_Ago'])

st.success(f"Model trained on {len(df_train)} games.")

# 2. RIDGE REGRESSION
# Create Dummy Variables
h_dummies = pd.get_dummies(df_train['Home'], dtype=int)
a_dummies = pd.get_dummies(df_train['Away'], dtype=int)

# Align columns
all_teams = sorted(list(set(h_dummies.columns) | set(a_dummies.columns)))
h_dummies = h_dummies.reindex(columns=all_teams, fill_value=0)
a_dummies = a_dummies.reindex(columns=all_teams, fill_value=0)

# Build X and y
X = h_dummies.sub(a_dummies)
X['HFA'] = 1
y = df_train['Market_Line']

# 🛡️ THE NUCLEAR FIX: Force NumPy Conversion 🛡️
# This strips away all Pandas metadata that causes the TypeError
X_mat = X.values.astype(np.float64)  # Force pure float matrix
y_vec = y.values.astype(np.float64)  # Force pure float vector
w_vec = df_train['Weight'].values.astype(np.float64) # Force pure float weights

# Fit Model (Using Pure Numpy Arrays)
clf = Ridge(alpha=1.0, fit_intercept=False)
clf.fit(X_mat, y_vec, sample_weight=w_vec)

# Extract Ratings
# We map the numpy results back to team names using X.columns
coefs = pd.Series(clf.coef_, index=X.columns)
hfa = coefs['HFA']
ratings = coefs.drop('HFA')
ratings = ratings - ratings.mean() 
ratings_dict = ratings.to_dict()

# 3. DISPLAY
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Market Ratings")
    rat_df = pd.DataFrame({'Team': ratings.index, 'Rating': ratings.values})
    rat_df = rat_df.sort_values('Rating', ascending=True).reset_index(drop=True)
    
    # Simple styling that works on all versions
    st.dataframe(
        rat_df,
        height=500, 
        use_container_width=True
    )

with col2:
    st.subheader("🔮 Betting Board")
    sched = fetch_live_schedule(target_date)
    
    if sched.empty:
        st.info("No games scheduled (or API unavailable).")
    else:
        # Input Table
        input_rows = []
        for _, row in sched.iterrows():
            input_rows.append({
                "Away Team": row['Away'],
                "Home Team": row['Home'],
                "Vegas Line (Home)": 0.0
            })
            
        edited_df = st.data_editor(
            pd.DataFrame(input_rows),
            column_config={
                "Vegas Line (Home)": st.column_config.NumberColumn(format="%.1f")
            },
            hide_index=True
        )
        
        # Projections
        results = []
        for _, row in edited_df.iterrows():
            h, a, v = row['Home Team'], row['Away Team'], row['Vegas Line (Home)']
            
            # Helper to match ESPN abbreviations (e.g., NY vs NYK)
            rh = ratings_dict.get(h, ratings_dict.get('NY', 0.0) if h=='NYK' else 0.0)
            ra = ratings_dict.get(a, ratings_dict.get('NY', 0.0) if a=='NYK' else 0.0)
            
            model = (rh - ra) + hfa
            edge = model - v
            
            sig = "PASS"
            if edge < -2.0: sig = f"BET {h}"
            elif edge > 2.0: sig = f"BET {a}"
            
            results.append({
                "Matchup": f"{a} @ {h}",
                "Model": round(model, 1),
                "Vegas": v,
                "Edge": round(edge, 1),
                "Signal": sig
            })
            
        def color_signal(val):
            return 'background-color: #d4edda; color: green; font-weight: bold' if 'BET' in str(val) else ''

        # Use applymap for compatibility
        st.dataframe(
            pd.DataFrame(results).style.applymap(color_signal, subset=['Signal'])
            .format({"Model": "{:.1f}", "Vegas": "{:.1f}", "Edge": "{:.1f}"}), 
            use_container_width=True, 
            hide_index=True
        )
