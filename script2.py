import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog
from sklearn.linear_model import Ridge
import time

# --- CONFIGURATION ---
CURRENT_SEASON = '2025-26'  # Format must be 'YYYY-YY'
SEASON_TYPE = 'Regular Season'

# TUNING: Recency Bias
# 0.05 = Slow decay (Long-term rating)
# 0.15 = Fast decay (Power Rankings style)
DECAY_ALPHA = 0.10

OUTPUT_FILE = "NBA_Market_Ratings.xlsx"

print(f"--- 🏀 NBA MARKET MODEL: {CURRENT_SEASON} ---")

# --- 1. FETCH DATA ---
print("Fetching Game Data from NBA API...")
# We use the official NBA API to get game IDs and scores
try:
    gamelog = leaguegamelog.LeagueGameLog(season=CURRENT_SEASON, season_type_all_star=SEASON_TYPE).get_data_frames()[0]
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# Filter for unique games (API returns 2 rows per game: one for Home, one for Away)
# We keep the row where the team is at HOME to simplify
games = gamelog[gamelog['MATCHUP'].str.contains(' vs. ')].copy()

# --- 2. PREP DATA WITH CLOSING LINES ---
# NOTE: The NBA API does not provide historical betting lines.
# In a real pro workflow, you would pull this from a paid odds API.
# For this script, we will SIMULATE closing lines using the actual Score Margin + Noise.
# *CRITICAL*: Replace this block with real odds data if you have a CSV of closing lines!

print("⚠️ Note: Using Implied Lines (Score Margin) for demonstration.")
print("   > In 'Pro' mode, you must merge external odds data here.")

matchups = []
unique_teams = set()

# Most recent game date for decay calc
last_date = pd.to_datetime(games['GAME_DATE']).max()

for _, row in games.iterrows():
    home_team = row['TEAM_ABBREVIATION']
    # Extract Away Team from Matchup string "BOS vs. MIA"
    away_team = row['MATCHUP'].split(' vs. ')[-1]

    unique_teams.add(home_team)
    unique_teams.add(away_team)

    # Calculate Margin (Home Points - Away Points)
    # Note: API row contains home stats. We need away score.
    # We can infer margin strictly from the Plus_Minus column in the API
    margin = row['PLUS_MINUS']

    # --- "THE TRUTH" (Replacing Real Odds) ---
    # Since we don't have a live Odds API key in this script, we assume the
    # Closing Line was roughly the Final Score but "smoother".
    # This effectively creates an "Efficiency Rating" model (like Net Rating).
    # IF YOU HAVE ODDS: closing_spread = row['CLOSING_SPREAD']
    implied_market_line = -margin  # Negative = Home Favored

    # Recency Weight
    game_date = pd.to_datetime(row['GAME_DATE'])
    days_ago = (last_date - game_date).days
    weight = np.exp(-DECAY_ALPHA * (days_ago / 7))  # Decay per week

    matchups.append({
        'Home': home_team,
        'Away': away_team,
        'Market_Line': implied_market_line,
        'Weight': weight,
        'Date': game_date
    })

df = pd.DataFrame(matchups)
print(f"Processed {len(df)} games.")

# --- 3. RIDGE REGRESSION ---
print("Running Regression...")

# Create Dummy Variables
h_dummies = pd.get_dummies(df['Home'], dtype=int)
a_dummies = pd.get_dummies(df['Away'], dtype=int)
all_teams = sorted(list(unique_teams))

# Reindex to ensure columns match
h_dummies = h_dummies.reindex(columns=all_teams, fill_value=0)
a_dummies = a_dummies.reindex(columns=all_teams, fill_value=0)

# X: Home - Away (1 for Home, -1 for Away)
X = h_dummies.sub(a_dummies)
X['HFA'] = 1  # Solve for Home Court Advantage constant
y = df['Market_Line']

# Ridge (Alpha 1.0 is standard for NBA to prevent overfitting on blowouts)
clf = Ridge(alpha=1.0, fit_intercept=False)
clf.fit(X, y, sample_weight=df['Weight'])

# Extract Ratings
coefs = pd.Series(clf.coef_, index=X.columns)
hfa_val = coefs['HFA']
team_ratings = coefs.drop('HFA')
team_ratings = team_ratings - team_ratings.mean()  # Normalize to 0

# --- 4. OUTPUT ---
ratings_df = pd.DataFrame({'Team': team_ratings.index, 'Rating': team_ratings.values})
ratings_df = ratings_df.sort_values('Rating', ascending=True).reset_index(drop=True)
# Note: Ascending=True because Negative Rating = Good (Favorite)

print(f"\n📊 Calculated Home Court Advantage: {hfa_val:.2f} points")
print("-" * 40)
print(ratings_df.head(10).to_string())

# --- 5. EXPORT ---
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    ratings_df.to_excel(writer, sheet_name='Power Ratings', index=False)
    df.to_excel(writer, sheet_name='Game Data Used', index=False)

print(f"\n✅ Done! Saved to {OUTPUT_FILE}")