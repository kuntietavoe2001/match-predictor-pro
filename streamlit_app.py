import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pickle
import os
import onnxruntime as ort
import pandas as pd
from datetime import datetime, date, timedelta

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Match Predictor Pro", page_icon="⚽", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Roboto:wght@400;500;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #4a0e1e 0%, #2d0a12 50%, #1a0609 100%);
    font-family: 'Roboto', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }

.main-header {
    background: linear-gradient(135deg, #6b1529 0%, #4a0e1e 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    border: 2px solid #8b1e3c;
    text-align: center;
}
.main-header h1 {
    color: white;
    font-family: 'Oswald', sans-serif;
    font-size: 2.5rem;
    font-weight: 500;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.main-header p {
    color: rgba(255,255,255,0.8);
    font-size: 1rem;
    margin-top: 0.5rem;
}

/* Form Elements */
.stSelectbox > div > div {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    color: white !important;
}
.stSelectbox > div > div:hover { border-color: #6b1529 !important; }
.stTextInput > div > div > input {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    color: white !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6b1529 !important;
    box-shadow: 0 0 0 2px rgba(107, 21, 41, 0.3) !important;
}

/* Date Input */
.stDateInput > div > div {
    background: #1e293b !important;
    border-radius: 8px !important;
}
.stDateInput > div > div > input {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    color: white !important;
}
.stDateInput > div > div > input:focus {
    border-color: #6b1529 !important;
    box-shadow: 0 0 0 2px rgba(107, 21, 41, 0.3) !important;
}
.stDateInput label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6b1529 0%, #4a0e1e 100%) !important;
    color: white !important;
    border: 1px solid #8b1e3c !important;
    border-radius: 5px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 400 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b1e3c 0%, #6b1529 100%) !important;
    box-shadow: 0 4px 15px rgba(107, 21, 41, 0.5) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    border: 1px solid #ef4444 !important;
    font-size: 1rem !important;
    padding: 1rem 2.5rem !important;
}

/* Labels */
.stSelectbox label, .stTextInput label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
h1, h2, h3, h4, h5, h6 {
    color: white !important;
    font-family: 'Oswald', sans-serif !important;
}
p, span, div { color: #e2e8f0; }

/* Expander */
.streamlit-expanderHeader {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #0f172a !important;
    border: 1px solid #475569 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%) !important;
    border-radius: 6px !important;
}
.stProgress > div > div {
    background: #1e293b !important;
    border-radius: 6px !important;
}

/* Login */
.login-container {
    display: flex; justify-content: center; align-items: center; min-height: 40vh;
}
.login-card {
    background: linear-gradient(145deg, #4a0e1e 0%, #2d0a12 100%);
    border: 2px solid #6b1529; border-radius: 12px; padding: 3rem;
    max-width: 420px; width: 70%;
    box-shadow: 0 25px 50px rgba(0,0,0,0.5);
}
.login-icon { font-size: 4rem; text-align: center; margin-bottom: 1.5rem; }
.login-title {
    font-family: 'Oswald', sans-serif; font-size: 2rem; font-weight: 700;
    text-align: center; color: white; text-transform: uppercase;
    letter-spacing: 2px; margin-bottom: 0.5rem;
}
.login-subtitle {
    font-size: 1rem; text-align: center; color: #94a3b8; margin-bottom: 2rem;
}

/* VS Badge */
.vs-badge {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    color: white; font-family: 'Oswald', sans-serif; font-size: 1.8rem;
    font-weight: 500; padding: 1rem 1.5rem; border-radius: 8px;
    display: inline-block; box-shadow: 0 8px 25px rgba(220, 38, 38, 0.4);
    text-transform: uppercase; letter-spacing: 2px;
}

/* Match Date Card */
.match-info-card {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #475569; border-radius: 12px; padding: 1.25rem;
    text-align: center;
}
.match-date-display {
    font-family: 'Oswald', sans-serif; font-size: 1.1rem;
    color: #dc2626; text-transform: uppercase; letter-spacing: 2px;
}
.match-day-display { font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem; }

/* Analysis Cards */
.analysis-card {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #475569; border-radius: 8px; padding: 1.5rem; margin: 0.5rem 0;
}
.analysis-title {
    font-family: 'Oswald', sans-serif; font-size: 1.1rem; font-weight: 600;
    color: #dc2626; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;
}
.analysis-item {
    display: flex; justify-content: space-between; padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.analysis-label { color: #94a3b8; }
.analysis-value { color: white; font-weight: 600; }

[data-testid="column"] { padding: 0.5rem; }
hr { border-color: rgba(255,255,255,0.1) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (all cached for performance)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_prediction_data():
    """Load player stats and injury data for predictions."""
    player_stats = pd.read_csv('dataset/epl_player_stats_24_25.csv')
    player_stats.columns = player_stats.columns.str.strip()
    if 'Rating' in player_stats.columns:
        player_stats['Rating'] = pd.to_numeric(player_stats['Rating'], errors='coerce').fillna(6.5)
    injuries = pd.read_csv('dataset/player_injuries_impact.csv')
    return player_stats, injuries


@st.cache_data
def load_app_data():
    """Load the pickled app data (teams, players, default features)."""
    with open("streamlit_app_data.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
@st.cache_resource
def load_model():
    model_path = "match_predictor_model.onnx"
    if os.path.exists(model_path):
        return ort.InferenceSession(model_path)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER & TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

PLAYER_STATS, INJURY_DATA = load_prediction_data()


def get_player_features(player_name: str, club: str) -> dict:
    """Get player features: rating, goals, assists, injury status."""
    features = {
        'rating': 6.5, 'goals': 0, 'assists': 0,
        'is_injured': 0, 'injury_severity': 0
    }
    if not player_name or pd.isna(player_name):
        return features

    player_row = PLAYER_STATS[
        (PLAYER_STATS['Player Name'].str.lower() == player_name.lower()) |
        (PLAYER_STATS['Player Name'].str.lower().str.contains(player_name.lower(), na=False))
    ]
    if len(player_row) > 0:
        row = player_row.iloc[0]
        features['rating'] = float(row.get('Rating', 6.5)) if pd.notna(row.get('Rating')) else 6.5
        features['goals'] = int(row.get('Goals', 0)) if pd.notna(row.get('Goals')) else 0
        features['assists'] = int(row.get('Assists', 0)) if pd.notna(row.get('Assists')) else 0

    injured = INJURY_DATA[INJURY_DATA['Name'].str.lower().str.contains(player_name.lower(), na=False)]
    if len(injured) > 0:
        features['is_injured'] = 1
        for col in ['Match1_missed_match_Result', 'Match2_missed_match_Result', 'Match3_missed_match_Result']:
            if col in injured.columns:
                features['injury_severity'] += injured[col].notna().sum()
    return features


def calculate_team_strength(lineup: dict, team_name: str) -> dict:
    """Calculate team strength metrics from lineup."""
    total_rating = total_goals = total_assists = 0
    injured_count = injury_severity = player_count = 0
    for pos, player in lineup.items():
        if player:
            f = get_player_features(player, team_name)
            total_rating += f['rating']
            total_goals += f['goals']
            total_assists += f['assists']
            injured_count += f['is_injured']
            injury_severity += f['injury_severity']
            player_count += 1
    if player_count == 0:
        player_count = 1
    return {
        'avg_rating': total_rating / player_count,
        'total_goals': total_goals, 'total_assists': total_assists,
        'injured_count': injured_count, 'injury_severity': injury_severity,
        'player_count': player_count
    }


def generate_match_features(h_strength: dict, a_strength: dict) -> np.ndarray:
    """Generate feature vector for match prediction."""
    features = np.zeros(20, dtype=np.float32)
    features[0] = (h_strength['avg_rating'] - a_strength['avg_rating']) / 2.0
    features[1] = h_strength['total_goals'] / 50.0
    features[2] = a_strength['total_goals'] / 50.0
    features[3] = h_strength['total_assists'] / 30.0
    features[4] = a_strength['total_assists'] / 30.0
    features[5] = -h_strength['injured_count'] * 0.1
    features[6] = -a_strength['injured_count'] * 0.1
    features[7] = -h_strength['injury_severity'] * 0.05
    features[8] = -a_strength['injury_severity'] * 0.05
    features[9] = 0.1
    features[10] = h_strength['avg_rating'] / 10.0
    features[11] = a_strength['avg_rating'] / 10.0
    features[12] = (h_strength['total_goals'] + h_strength['total_assists']) / 60.0
    features[13] = (a_strength['total_goals'] + a_strength['total_assists']) / 60.0
    features[14:] = 0.5
    return features.reshape(1, -1)


def calculate_schedule_factor(match_date: date) -> dict:
    """Calculate scheduling factors that influence match prediction."""
    day_of_week = match_date.weekday()
    is_midweek = day_of_week in [1, 2, 3]
    fatigue_factor = 0.03 if is_midweek else 0.0

    month = match_date.month
    if month == 12:
        congestion_factor = 0.05
    elif month in [1, 2]:
        congestion_factor = 0.03
    elif month in [4, 5]:
        congestion_factor = 0.02
    else:
        congestion_factor = 0.0

    season_start = date(match_date.year if match_date.month >= 8 else match_date.year - 1, 8, 1)
    days_into_season = (match_date - season_start).days
    season_progress = min(1.0, max(0.0, days_into_season / 300.0))

    return {
        'is_midweek': is_midweek,
        'fatigue_factor': fatigue_factor,
        'congestion_factor': congestion_factor,
        'season_progress': season_progress,
        'day_name': match_date.strftime('%A'),
        'date_display': match_date.strftime('%d %B %Y')
    }


def predict_goal_scorers(lineup: dict, team_name: str, num_goals: int) -> list:
    """Predict which players scored based on their goal-scoring probability."""
    import random
    random.seed(hash(team_name) % (2**32))
    scorers = []
    if num_goals <= 0:
        return scorers

    forward_positions = ['LW', 'ST', 'RW', 'ST1', 'ST2']
    mid_positions = ['CM1', 'CM2', 'CM3', 'LM', 'RM', 'DM1', 'DM2', 'AM']
    player_scores = []
    for pos, player in lineup.items():
        if not player:
            continue
        f = get_player_features(player, team_name)
        pos_weight = 2.0 if pos in forward_positions else (0.3 if pos in mid_positions else 0.1)
        score_prob = min(0.8, (f['goals'] / 20.0) * pos_weight * (f['rating'] / 7.0))
        player_scores.append((player, score_prob, f['goals']))
    player_scores.sort(key=lambda x: x[1], reverse=True)

    goals_remaining = num_goals
    for player, prob, _ in player_scores:
        if goals_remaining <= 0:
            break
        if random.random() < prob * 1.5 or (goals_remaining > 0 and prob > 0.3):
            scorers.append(player)
            goals_remaining -= 1
    while goals_remaining > 0 and player_scores:
        player, _, _ = player_scores[min(goals_remaining, len(player_scores) - 1)]
        if player not in scorers:
            scorers.append(player)
            goals_remaining -= 1
    return scorers[:num_goals]


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATIONS & LINEUP MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_EMAIL = "user@demo.com"
DEMO_PASSWORD = "demo123"

def check_login(email, password):
    return email.strip() == DEMO_EMAIL and password == DEMO_PASSWORD

FORMATION_POSITIONS = {
    "4-3-3": [["GK"], ["LB", "CB1", "CB2", "RB"], ["CM1", "CM2", "CM3"], ["LW", "ST", "RW"]],
    "4-4-2": [["GK"], ["LB", "CB1", "CB2", "RB"], ["LM", "CM1", "CM2", "RM"], ["ST1", "ST2"]],
    "4-2-3-1": [["GK"], ["LB", "CB1", "CB2", "RB"], ["DM1", "DM2"], ["LW", "AM", "RW"], ["ST"]],
}

POS_MAP = {
    "GK": "GKP", "LB": "DEF", "CB1": "DEF", "CB2": "DEF", "RB": "DEF",
    "CM1": "MID", "CM2": "MID", "CM3": "MID", "LM": "MID", "RM": "MID",
    "DM1": "MID", "DM2": "MID", "AM": "MID",
    "LW": "FWD", "ST": "FWD", "RW": "FWD", "ST1": "FWD", "ST2": "FWD"
}


def auto_fill_lineup(formation: str, team_players_dict: dict) -> dict:
    """Auto-fill lineup with first 11 eligible players in correct positions."""
    formation_rows = FORMATION_POSITIONS[formation]
    selected_lineup = {}
    used_players = set()
    valid_players = {k: v for k, v in team_players_dict.items() if k and k.strip()}

    all_positions = [pos for row in formation_rows for pos in row]
    for pos_key in all_positions:
        required_pos = POS_MAP.get(pos_key, "")
        eligible = [(p, pos) for p, pos in valid_players.items()
                    if pos == required_pos and p not in used_players]
        if eligible:
            selected_player = sorted(eligible, key=lambda x: x[0])[0][0]
            selected_lineup[pos_key] = selected_player
            used_players.add(selected_player)
        else:
            available = [p for p in valid_players.keys() if p not in used_players]
            if available:
                player = sorted(available)[0]
                selected_lineup[pos_key] = player
                used_players.add(player)
            else:
                selected_lineup[pos_key] = ""
    return selected_lineup


def get_player_short_name(full_name: str) -> str:
    if not full_name:
        return ""
    parts = full_name.split()
    return parts[-1].upper()[:10] if len(parts) >= 2 else full_name.upper()[:10]


# ═══════════════════════════════════════════════════════════════════════════════
# STARTING XI VISUALIZATION (broadcast-style)
# ═══════════════════════════════════════════════════════════════════════════════

# Shared CSS for components.html — defined once, used for both team panels
STARTING_XI_CSS = """
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Roboto',sans-serif;background:transparent;}
.outfield{background:linear-gradient(180deg,#2563eb,#1d4ed8);clip-path:polygon(20% 0%,80% 0%,100% 15%,100% 100%,0% 100%,0% 15%);box-shadow:0 2px 8px rgba(0,0,0,0.4);}
.goalkeeper{background:linear-gradient(180deg,#eab308,#ca8a04);clip-path:polygon(20% 0%,80% 0%,100% 15%,100% 100%,0% 100%,0% 15%);box-shadow:0 2px 8px rgba(0,0,0,0.4);}
.pitch{background:linear-gradient(180deg,#1a6b32 0%,#1f7a3a 10%,#1a6b32 20%,#1f7a3a 30%,#1a6b32 40%,#1f7a3a 50%,#1a6b32 60%,#1f7a3a 70%,#1a6b32 80%,#1f7a3a 90%,#1a6b32 100%);border-radius:8px;position:relative;min-height:300px;border:3px solid rgba(255,255,255,0.3);box-shadow:inset 0 0 30px rgba(0,0,0,0.4);}
.pitch::before{content:'';position:absolute;top:50%;left:5%;right:5%;height:2px;background:rgba(255,255,255,0.4);}
.pitch::after{content:'';position:absolute;top:calc(50% - 35px);left:50%;transform:translateX(-50%);width:70px;height:70px;border:2px solid rgba(255,255,255,0.4);border-radius:50%;}
"""


def generate_pitch_positions(formation: str, lineup: dict) -> str:
    """Generate HTML for player positions on the pitch."""
    position_coords = {
        "4-3-3": {
            "GK": (50, 90), "LB": (15, 70), "CB1": (35, 72), "CB2": (65, 72), "RB": (85, 70),
            "CM1": (25, 45), "CM2": (50, 50), "CM3": (75, 45),
            "LW": (15, 20), "ST": (50, 15), "RW": (85, 20)
        },
        "4-4-2": {
            "GK": (50, 90), "LB": (15, 70), "CB1": (35, 72), "CB2": (65, 72), "RB": (85, 70),
            "LM": (15, 45), "CM1": (38, 48), "CM2": (62, 48), "RM": (85, 45),
            "ST1": (35, 18), "ST2": (65, 18)
        },
        "4-2-3-1": {
            "GK": (50, 90), "LB": (15, 70), "CB1": (35, 72), "CB2": (65, 72), "RB": (85, 70),
            "DM1": (35, 52), "DM2": (65, 52),
            "LW": (15, 32), "AM": (50, 35), "RW": (85, 32), "ST": (50, 12)
        }
    }
    coords = position_coords.get(formation, position_coords["4-3-3"])
    parts = ['<div style="position:relative;width:100%;height:280px;">']
    for pos, (x, y) in coords.items():
        player = lineup.get(pos, "")
        short_name = get_player_short_name(player)
        jersey_class = "goalkeeper" if pos == "GK" else "outfield"
        parts.append(f'''
        <div style="position:absolute;left:{x}%;top:{y}%;transform:translate(-50%,-50%);">
            <div style="display:flex;flex-direction:column;align-items:center;gap:3px;text-align:center;">
                <div class="{jersey_class}" style="width:32px;height:32px;"></div>
                <div style="font-size:0.6rem;font-weight:600;color:white;text-shadow:1px 1px 2px rgba(0,0,0,0.8);max-width:65px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-transform:uppercase;">{short_name}</div>
            </div>
        </div>''')
    parts.append('</div>')
    return ''.join(parts)


def render_starting_xi_display(team_name: str, formation: str, lineup: dict, side: str):
    """Render the broadcast-style Starting XI display."""
    formation_rows = FORMATION_POSITIONS[formation]
    ordered_players = [(pos, lineup[pos]) for row in formation_rows for pos in row
                       if pos in lineup and lineup[pos]]

    pitch_html = generate_pitch_positions(formation, lineup)

    player_list_html = ""
    for idx, (pos, player) in enumerate(ordered_players, 1):
        short_name = get_player_short_name(player)
        player_list_html += f'''
        <div style="display:flex;align-items:center;margin:3px 0;">
            <div style="background:linear-gradient(135deg,#1e293b,#0f172a);color:#94a3b8;font-family:'Oswald',sans-serif;font-size:0.85rem;font-weight:600;min-width:28px;height:26px;display:flex;align-items:center;justify-content:center;border-radius:3px 0 0 3px;">{idx:02d}</div>
            <div style="flex:1;background:linear-gradient(135deg,#475569 0%,#334155 40%,#1e293b 100%);height:26px;display:flex;align-items:center;padding:0 10px;position:relative;clip-path:polygon(0 0,100% 0,95% 100%,0% 100%);">
                <span style="font-family:'Roboto',sans-serif;font-size:0.75rem;font-weight:500;color:white;text-transform:uppercase;letter-spacing:0.5px;position:relative;z-index:1;">{short_name}</span>
            </div>
        </div>'''

    full_html = f'''<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<style>{STARTING_XI_CSS}</style>
</head><body>
<div style="background:linear-gradient(145deg,#4a0e1e,#2d0a12);border:2px solid #6b1529;border-radius:12px;padding:1.5rem;overflow:hidden;">
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:260px;">
            <div style="font-family:'Oswald',sans-serif;font-size:1.6rem;font-weight:700;color:white;text-transform:uppercase;letter-spacing:3px;margin-bottom:1rem;text-shadow:2px 2px 4px rgba(0,0,0,0.5);">STARTING XI</div>
            <div style="background:linear-gradient(135deg,#1e293b,#334155);border:2px solid #475569;border-radius:30px 0 0 30px;padding:0.6rem 1.5rem 0.6rem 2.5rem;display:inline-flex;align-items:center;margin-bottom:1rem;position:relative;">
                <span style="font-family:'Oswald',sans-serif;font-size:1.2rem;font-weight:600;color:white;text-transform:uppercase;letter-spacing:1px;">{team_name}</span>
            </div>
            <div class="pitch">{pitch_html}</div>
        </div>
        <div style="flex:1;min-width:180px;display:flex;flex-direction:column;justify-content:center;">
            {player_list_html}
        </div>
    </div>
</div>
</body></html>'''

    components.html(full_html, height=450, scrolling=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def render_team_panel(side: str, teams: list, players_data: dict):
    """Render team selection + lineup + Starting XI panel."""
    side_key = side.lower()

    team = st.selectbox(f"Select {side} Team", teams, key=f"{side_key}_t")
    formation = st.selectbox(f"{side} Formation", list(FORMATION_POSITIONS.keys()), key=f"{side_key}_f")
    team_players_dict = players_data.get(team, {})
    full_roster = sorted([p for p in team_players_dict.keys() if p])

    lineup_key = f"{side_key}_lineup"
    prev_team = st.session_state.get(f"{side_key}_prev_team")
    prev_formation = st.session_state.get(f"{side_key}_prev_formation")
    if lineup_key not in st.session_state or prev_team != team or prev_formation != formation:
        st.session_state[lineup_key] = auto_fill_lineup(formation, team_players_dict)
    st.session_state[f"{side_key}_prev_team"] = team
    st.session_state[f"{side_key}_prev_formation"] = formation

    if st.button("Auto-fill Lineup", key=f"{side_key}_autofill"):
        st.session_state[lineup_key] = auto_fill_lineup(formation, team_players_dict)
        st.rerun()

    with st.expander("Edit Lineup", expanded=False):
        selected_lineup = {}
        used_players = set()
        for row in FORMATION_POSITIONS[formation]:
            cols = st.columns(len(row))
            for i, pos_key in enumerate(row):
                with cols[i]:
                    required_pos = POS_MAP.get(pos_key, "")
                    eligible = [p for p, pos in team_players_dict.items()
                               if pos == required_pos and p not in used_players]
                    options = sorted(eligible) if eligible else full_roster
                    current = st.session_state[lineup_key]
                    default_idx = (options.index(current[pos_key])
                                   if pos_key in current and current[pos_key] in options else 0)
                    selected_lineup[pos_key] = st.selectbox(
                        pos_key, options, index=default_idx, key=f"{side_key}_{pos_key}")
                    used_players.add(selected_lineup[pos_key])
        st.session_state[lineup_key] = selected_lineup

    selected_lineup = st.session_state[lineup_key]
    render_starting_xi_display(team, formation, selected_lineup, side)
    return team, selected_lineup


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE & LOGIN
# ═══════════════════════════════════════════════════════════════════════════════

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("""
    <div class="login-container">
        <div class="login-card">
            <div class="login-icon">⚽</div>
            <div class="login-title">Match Predictor</div>
            <div class="login-subtitle">Sign in to access match predictions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        if st.button("Sign In", use_container_width=True):
            if check_login(email, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials. Try user@demo.com / demo123")
        st.markdown("""
        <div style="text-align:center;margin-top:1.5rem;padding:1rem;background:rgba(255,255,255,0.05);border-radius:8px;border:1px solid rgba(107,21,41,0.3);">
            <p style="color:#94a3b8;font-size:0.9rem;margin:0;">Demo Credentials</p>
            <p style="color:#dc2626;font-size:0.85rem;margin:0.5rem 0 0 0;">user@demo.com / demo123</p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>⚽ Match Predictor Pro</h1>
    <p>Football Match Predictions with Player Analysis</p>
</div>
""", unsafe_allow_html=True)

data = load_app_data()
model = load_model()

# ── Match Date Selection ──────────────────────────────────────────────────────

st.markdown("### 📅 Match Details")
date_col1, date_col2, date_col3 = st.columns([2, 2, 1])

with date_col1:
    match_date = st.date_input(
        "Match Date",
        value=date.today(),
        min_value=date(2008, 8, 1),
        max_value=date(2026, 12, 31),
        key="match_date",
        help="Select the match date. Affects predictions via schedule congestion, midweek fatigue, and season factors."
    )

schedule_info = calculate_schedule_factor(match_date)

with date_col2:
    congestion_note = " · ⚠️ Congestion period" if schedule_info['congestion_factor'] > 0.03 else ""
    st.markdown(f"""
    <div class="match-info-card">
        <div class="match-date-display">{schedule_info['date_display']}</div>
        <div class="match-day-display">
            {schedule_info['day_name']}
            {'· Midweek Fixture' if schedule_info['is_midweek'] else '· Weekend Fixture'}
        </div>
        <div style="margin-top:0.5rem;font-size:0.8rem;color:#64748b;">
            Season progress: {schedule_info['season_progress']*100:.0f}%{congestion_note}
        </div>
    </div>
    """, unsafe_allow_html=True)

with date_col3:
    fatigue_val = schedule_info['fatigue_factor'] + schedule_info['congestion_factor']
    st.markdown(f"""
    <div style="background:linear-gradient(145deg,#1e293b,#0f172a);border:1px solid #475569;border-radius:12px;padding:1rem;text-align:center;margin-top:0.5rem;">
        <div style="font-family:'Oswald',sans-serif;color:#dc2626;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Fatigue</div>
        <div style="color:white;font-size:1.5rem;font-weight:700;font-family:'Oswald',sans-serif;">
            {'+' if fatigue_val > 0 else ''}{fatigue_val*100:.0f}%
        </div>
        <div style="font-size:0.7rem;color:#64748b;">Schedule impact</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Team Selection Panels ─────────────────────────────────────────────────────

left, right = st.columns(2)
with left:
    h_team, h_lineup = render_team_panel("Home", data['teams'], data['players_info'])
with right:
    a_team, a_lineup = render_team_panel(
        "Away", [t for t in data['teams'] if t != h_team], data['players_info'])

st.markdown('<div style="text-align:center;margin:2rem 0;"><span class="vs-badge">VS</span></div>',
            unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────

if st.button("PREDICT MATCH RESULT", type="primary", use_container_width=True):
    if model is None:
        st.error("Model file not found! Please ensure match_predictor_model.keras is in the app directory.")
    else:
        h_strength = calculate_team_strength(h_lineup, h_team)
        a_strength = calculate_team_strength(a_lineup, a_team)

        # Team Analysis
        with st.expander("📊 Team Analysis", expanded=True):
            col1, col2 = st.columns(2)
            for col, tname, strength, icon in [
                (col1, h_team, h_strength, "🏠"), (col2, a_team, a_strength, "✈️")
            ]:
                with col:
                    inj_color = '#ef4444' if strength['injured_count'] > 0 else '#10b981'
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div class="analysis-title">{icon} {tname}</div>
                        <div class="analysis-item"><span class="analysis-label">Average Rating</span><span class="analysis-value">{strength['avg_rating']:.2f}</span></div>
                        <div class="analysis-item"><span class="analysis-label">Total Goals</span><span class="analysis-value">{strength['total_goals']}</span></div>
                        <div class="analysis-item"><span class="analysis-label">Total Assists</span><span class="analysis-value">{strength['total_assists']}</span></div>
                        <div class="analysis-item"><span class="analysis-label">Injured Players</span><span class="analysis-value" style="color:{inj_color};">{strength['injured_count']}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

        # Model prediction
        match_features = generate_match_features(h_strength, a_strength)
        default_features = data['default_features']
        if default_features.shape[1] > match_features.shape[1]:
            padded = np.zeros((1, default_features.shape[1]), dtype=np.float32)
            padded[0, :match_features.shape[1]] = match_features[0]
            padded[0, match_features.shape[1]:] = default_features[0, match_features.shape[1]:]
        else:
            padded = match_features

        base_prob = float(model.predict(padded, verbose=0).flatten()[0])

        # Final probability with team strength + schedule factors
        rating_diff = (h_strength['avg_rating'] - a_strength['avg_rating']) / 10.0
        h_attack = (h_strength['total_goals'] + h_strength['total_assists']) / 60.0
        a_attack = (a_strength['total_goals'] + a_strength['total_assists']) / 60.0
        attack_diff = h_attack - a_attack
        injury_impact = ((a_strength['injured_count'] * 0.1 + a_strength['injury_severity'] * 0.05) -
                         (h_strength['injured_count'] * 0.1 + h_strength['injury_severity'] * 0.05))

        sched = calculate_schedule_factor(match_date)
        prob = (base_prob * 0.3 + 0.35 +
                rating_diff * 0.2 + attack_diff * 0.1 +
                injury_impact + 0.08 -
                sched['fatigue_factor'] * 0.5 -
                sched['congestion_factor'] * 0.3)
        prob = max(0.05, min(0.95, prob))

        # Outcome & scores
        if prob >= 0.75:
            outcome = "Home Win"
            h_s = min(4, int(2 + h_attack * 3))
            a_s = min(3, int(1 + a_attack * 2 + (1 if prob < 0.85 else 0)))
        elif prob >= 0.5:
            outcome = "Home Win" if prob > 0.6 else "Draw"
            h_s = min(3, int(1 + h_attack * 2))
            a_s = min(2, int(1 + a_attack * 2))
        elif prob >= 0.35:
            outcome, h_s, a_s = "Draw", 1, 1
        elif prob >= 0.2:
            outcome = "Away Win"
            h_s = min(2, int(1 + h_attack))
            a_s = min(3, int(2 + a_attack * 2))
        else:
            outcome = "Away Win"
            h_s = min(1, int(h_attack * 2))
            a_s = min(4, int(2 + a_attack * 3))

        # Goal scorers & times
        h_scorers = predict_goal_scorers(h_lineup, h_team, h_s)
        a_scorers = predict_goal_scorers(a_lineup, a_team, a_s)

        import random
        random.seed(hash(h_team + a_team) % (2**32))

        def gen_times(n):
            times = sorted([random.randint(1, 90) + (random.choice([0,1,2,3,4]) if random.random() < 0.15 else 0) for _ in range(n)])
            return [f"{t}'" if t <= 90 else f"90'+{t-90}'" for t in times]

        h_times = gen_times(len(h_scorers)) if h_scorers else []
        a_times = gen_times(len(a_scorers)) if a_scorers else []

        def fmt_scorers(scorers, times):
            if not scorers: return ""
            texts = []
            for i, s in enumerate(scorers):
                t = times[i] if i < len(times) else ""
                name = s.split()[-1] if s.split() else s
                texts.append(f"{name} {t}")
            if len(texts) > 3:
                return f"{', '.join(texts[:2])} <span style='color:#3b82f6;cursor:pointer;'>+{len(texts)-2} more</span>"
            return ", ".join(texts)

        h_scorers_text = fmt_scorers(h_scorers, h_times)
        a_scorers_text = fmt_scorers(a_scorers, a_times)

        # Win probabilities
        h_wp = prob * 100 if outcome == "Home Win" else (100 - prob * 100) / 2 if outcome == "Draw" else (1 - prob) * 30
        a_wp = (1 - prob) * 100 if outcome == "Away Win" else (100 - prob * 100) / 2 if outcome == "Draw" else prob * 30
        d_wp = 100 - h_wp - a_wp
        total = h_wp + d_wp + a_wp
        h_wp, d_wp, a_wp = (h_wp/total)*100, (d_wp/total)*100, (a_wp/total)*100

        midweek_badge = '<span style="display:inline-block;background:#475569;color:#e2e8f0;padding:0.2rem 0.6rem;border-radius:4px;font-size:0.7rem;margin-left:0.5rem;">MIDWEEK</span>' if sched['is_midweek'] else ''

        result_html = f"""
        <div style="background:linear-gradient(145deg,#f8fafc 0%,#e2e8f0 100%);border-radius:16px;padding:2rem;margin:2rem 0;box-shadow:0 4px 20px rgba(0,0,0,0.1);">
            <div style="text-align:center;margin-bottom:1rem;">
                <span style="font-size:0.85rem;color:#6b7280;">{sched['date_display']} · {sched['day_name']}{midweek_badge}</span>
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
                <div style="display:flex;align-items:center;gap:1rem;flex:1;">
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                        <span style="color:#374151;font-size:0.7rem;">▶</span>
                        <div style="width:50px;height:50px;background:linear-gradient(135deg,#1e3a8a,#3b82f6);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:0.8rem;box-shadow:0 2px 8px rgba(59,130,246,0.4);">{h_team[:3].upper()}</div>
                    </div>
                    <span style="font-family:'Inter',sans-serif;font-size:1.4rem;font-weight:600;color:#1f2937;">{h_team}</span>
                </div>
                <div style="text-align:center;padding:0 2rem;">
                    <div style="font-family:'Inter',sans-serif;font-size:4rem;font-weight:800;color:#111827;letter-spacing:-2px;">
                        {h_s} <span style="color:#9ca3af;font-weight:400;">-</span> {a_s}
                    </div>
                    <div style="font-size:0.85rem;color:#6b7280;margin-top:-0.5rem;">Match Prediction</div>
                </div>
                <div style="display:flex;align-items:center;gap:1rem;flex:1;justify-content:flex-end;">
                    <span style="font-family:'Inter',sans-serif;font-size:1.4rem;font-weight:600;color:#1f2937;">{a_team}</span>
                    <div style="width:50px;height:50px;background:linear-gradient(135deg,#7c3aed,#a78bfa);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:0.8rem;box-shadow:0 2px 8px rgba(124,58,237,0.4);">{a_team[:3].upper()}</div>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:flex-start;padding:1rem 0;border-top:1px solid #e5e7eb;margin-top:0.5rem;">
                <div style="flex:1;text-align:left;"><span style="font-size:0.9rem;color:#4b5563;">{h_scorers_text}</span></div>
                <div style="text-align:center;padding:0 1rem;"><span style="font-size:1.2rem;color:#9ca3af;">⚽</span></div>
                <div style="flex:1;text-align:right;"><span style="font-size:0.9rem;color:#4b5563;">{a_scorers_text}</span></div>
            </div>
            <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e5e7eb;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                    <span style="font-size:1rem;font-weight:600;color:#3b82f6;">{h_wp:.1f}%</span>
                    <span style="font-size:0.85rem;color:#6b7280;">Win probability</span>
                    <span style="font-size:1rem;font-weight:600;color:#6b7280;">{a_wp:.1f}%</span>
                </div>
                <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;background:#e5e7eb;">
                    <div style="width:{h_wp}%;background:linear-gradient(90deg,#3b82f6,#60a5fa);"></div>
                    <div style="width:{d_wp}%;background:#d1d5db;"></div>
                    <div style="width:{a_wp}%;background:linear-gradient(90deg,#9ca3af,#6b7280);"></div>
                </div>
            </div>
            <div style="text-align:center;margin-top:1.5rem;">
                <span style="display:inline-block;background:linear-gradient(135deg,#10b981,#059669);color:white;padding:0.6rem 1.5rem;border-radius:30px;font-weight:600;font-size:0.95rem;box-shadow:0 4px 15px rgba(16,185,129,0.4);">
                    Predicted: {outcome}
                </span>
            </div>
        </div>
        """
        components.html(result_html, height=420, scrolling=False)
