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
    decay_alpha = st.slider("Recency Decay", 0.0, 0.1, 0.035, step=0.001)
    target_date = st.date_input("Target Date", value=datetime.now())
    
    st.divider()
    st.subheader("Data Source")
    data_source = st.radio("Choose Source:", ["Auto-Web (Last 21 Days)", "Upload CSV (Pro)"])
    
    csv_file = None
    if data_source == "Upload CSV (Pro)":
        csv_file = st.file_uploader("Upload Game Data", type=["csv"])
        st.caption("Required cols: Date, Home, Away, Market_Line")

# --- HELPERS ---

@st.cache_data(ttl=3600)
def fetch_recent_market_data(days_back=21):
    """
    Scrapes ONLY the last 21 days from ESPN to avoid Timeouts.
    This provides just enough REAL data to generate ratings without a CSV.
    """
    games = []
    base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    
    # Generate dates
    dates = [datetime.now() - timedelta(days=x) for x in range(1, days_back + 1)]
    
    # We use a progress bar so you know it's working
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, d in enumerate(dates):
        pct = (i + 1) / len(dates)
        progress_bar.progress(pct)
        
        date_str = d.strftime("%Y%m%d")
        try:
            r = requests.get(f"{base_url}?dates={date_str}", timeout=1.0)
            data = r.json()
            
            if 'events' not in data: continue
            
            for event in data['events']:
                comp = event['competitions'][0]
                competitors = comp['competitors']
                home = next(filter(lambda x: x['homeAway'] == 'home', competitors))
                away = next(filter(lambda x: x['homeAway'] == 'away', competitors))
                
                h_abbr = home['team']['abbreviation']
                a_abbr = away['team']['abbreviation']
                
                # PARSE ODDS
                # We strictly look for the "Odds" tag. If missing, we skip the game.
                spread_val = 0.0
                if 'odds' in comp and len(comp['odds']) > 0:
                    details = comp['odds'][0].get('details', 'EVEN')
                    if details != 'EVEN':
                        try:
                            parts = details.split(' ')
                            fav_team = parts[0]
                            line_val = float(parts[1])
                            
                            if fav_team == h_abbr: spread_val = line_val
                            elif fav_team == a_abbr: spread_val = -line_val
                        except: continue
                
                if spread_val != 0.0:
                    games.append({
                        'Date': d,
                        'Home': h_abbr,
                        'Away': a_abbr,
                        'Market_Line': spread_val
                    })
        except:
            continue
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(games)

def load_data(source_type, uploaded_file):
    if source_type == "Upload CSV (Pro)":
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Normalize column names if needed
                return df
            except Exception as e:
                st.error(f"Error: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame() # Return empty if no file
            
    elif source_type == "Auto-Web (Last 21 Days)":
        return fetch_recent_market_data()
        
    return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_live_schedule(t_date):
    """Fetches today's schedule + opening lines."""
    date_str = t_date.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    matchups = []
    try:
        r = requests.get(url, timeout=3)
        data = r.json()
        if 'events' in data:
            for event in data['events']:
                comp = event['competitions'][0]
                h = next(filter(lambda x: x['homeAway'] == 'home', comp['competitors']))
                a = next(filter(lambda x: x['homeAway'] == 'away', comp['competitors']))
                
                # Get Line
                vegas = 0.0
                if 'odds' in comp and len(comp['odds']) > 0:
                    details = comp['odds'][0].get('details', 'EVEN')
                    if details != 'EVEN':
                        try:
                            parts = details.split(' ')
                            if parts[0] == h['team']['abbreviation']: vegas = float(parts[1])
                            elif parts[0] == a['team']['abbreviation']: vegas = -float(parts[1])
                        except: pass
                        
                matchups.append({
                    'Home': h['team']['abbreviation'],
                    'Away': a['team']['abbreviation'],
                    'Vegas': vegas
                })
    except: pass
    return pd.DataFrame(matchups)

# --- MAIN LOGIC ---

st.subheader("1. Data Loading")
df_train = load_data(data_source, csv_file)

if df_train.empty:
    st.info("👋 Waiting for data. Please select 'Auto-Web' or Upload a CSV to generate ratings.")
    st.stop() # HALT APP HERE if no data

# Data Processing
st.success(f"Loaded {len(df_train)} games of historical odds.")
df_train['Date'] = pd.to_datetime(df_train['Date'])
last_date = df_train['Date'].max()
df_train['Days_Ago'] = (last_date - df_train['Date']).dt.days
df_train['Weight'] = np.exp(-decay_alpha * df_train['Days_Ago'])

# 2. RIDGE REGRESSION
h_dummies = pd.get_dummies(df_train['Home'], dtype=int)
a_dummies = pd.get_dummies(df_train['Away'], dtype=int)
all_teams = sorted(list(set(h_dummies.columns) | set(a_dummies.columns)))

h_dummies = h_dummies.reindex(columns=all_teams, fill_value=0)
a_dummies = a_dummies.reindex(columns=all_teams, fill_value=0)

X = h_dummies.sub(a_dummies)
X['HFA'] = 1
y = df_train['Market_Line']

# Strict Float Conversion
X_mat = X.values.astype(np.float64)
y_vec = y.values.astype(np.float64)
w_vec = df_train['Weight'].values.astype(np.float64)

clf = Ridge(alpha=0.5, fit_intercept=False)
clf.fit(X_mat, y_vec, sample_weight=w_vec)

coefs = pd.Series(clf.coef_, index=X.columns)
hfa = coefs['HFA']
ratings = coefs.drop('HFA')
ratings = ratings - ratings.mean() 
ratings_dict = ratings.to_dict()

# 3. DISPLAY
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Pure Market Ratings")
    st.caption(f"Implied HFA: {hfa:.2f} pts")
    rat_df = pd.DataFrame({'Team': ratings.index, 'Rating': ratings.values})
    rat_df = rat_df.sort_values('Rating', ascending=True).reset_index(drop=True)
    st.dataframe(rat_df.style.background_gradient(cmap="RdYlGn_r", subset=['Rating']).format({"Rating": "{:.2f}"}), use_container_width=True, height=500)

with col2:
    st.subheader(f"🔮 Betting Board ({target_date})")
    sched = fetch_live_schedule(target_date)
    
    if sched.empty:
        st.warning("No games found today.")
    else:
        # Input Table
        input_rows = []
        for _, row in sched.iterrows():
            input_rows.append({
                "Away Team": row['Away'],
                "Home Team": row['Home'],
                "Vegas Line (Home)": row['Vegas']
            })
            
        edited_df = st.data_editor(pd.DataFrame(input_rows), column_config={"Vegas Line (Home)": st.column_config.NumberColumn(format="%.1f")}, hide_index=True)
        
        results = []
        for _, row in edited_df.iterrows():
            h, a, v = row['Home Team'], row['Away Team'], row['Vegas Line (Home)']
            # Handle mismatches safely
            rh = ratings_dict.get(h, 0.0)
            ra = ratings_dict.get(a, 0.0)
            
            model = (rh - ra) + hfa
            edge = model - v
            
            sig = "PASS"
            if edge < -2.0: sig = f"BET {h}"
            elif edge > 2.0: sig = f"BET {a}"
            
            results.append({"Matchup": f"{a} @ {h}", "Model": model, "Vegas": v, "Edge": edge, "Signal": sig})
            
        def color_signal(val): return 'background-color: #d4edda; color: green; font-weight: bold' if 'BET' in str(val) else ''
        st.dataframe(pd.DataFrame(results).style.applymap(color_signal, subset=['Signal']).format({"Model": "{:.1f}", "Vegas": "{:.1f}", "Edge": "{:.1f}"}), use_container_width=True, hide_index=True)
