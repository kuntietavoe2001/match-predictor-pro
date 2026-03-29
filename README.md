# ⚽ Match Predictor Pro

**EPL Football Match Prediction App** — A Streamlit application that uses a Keras deep learning model trained on 6,300+ historical EPL match statistics to predict match outcomes, goal scorers, and win probabilities.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit&logoColor=white)

---

## Features

- **Match Prediction** — Predicts match outcome (Home Win / Draw / Away Win) with win probabilities
- **Goal Scorer Prediction** — Identifies likely goal scorers based on player stats and position weighting
- **Match Date Selection** — Schedule-aware predictions factoring in midweek fatigue, fixture congestion, and season progression
- **Broadcast-Style Starting XI** — Interactive football pitch visualization with jersey icons and player list
- **Team Strength Analysis** — Player ratings, goals, assists, and injury impact analysis
- **Formation Tactics** — Support for 4-3-3, 4-4-2, and 4-2-3-1 formations with auto-fill lineups
- **21 EPL Teams** — Full 2024/25 season squad data with 560+ players

## Demo Credentials

```
Email:    user@demo.com
Password: demo123
```

---

## Project Structure

```
match-predictor-pro/
├── streamlit_app.py                    # Main application (907 lines)
├── requirements.txt                    # Python dependencies
├── match_predictor_model.onnx          # Trained model (ONNX format — Python 3.11-3.14+ compatible)
├── streamlit_app_data.pkl              # Pickled app data (teams, players, features)
├── .streamlit/
│   └── config.toml                     # Streamlit theme configuration
├── dataset/
│   ├── match statistics.csv            # 6,340 historical EPL matches (2008-2026)
│   ├── epl_player_stats_24_25.csv      # 562 player stats for 2024/25 season
│   └── player_injuries_impact.csv      # 656 player injury impact records
├── saved_model_match_predictor/        # SavedModel format (backup)
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   ├── variables/
│   │   ├── variables.data-00000-of-00001
│   │   └── variables.index
│   └── assets/
├── exploratory_analysis_(4)_(1).ipynb  # EDA & model training notebook
├── .gitignore
└── README.md
```

---

## Deployment Guide

### Option 1: Deploy to Streamlit Community Cloud (Recommended)

#### Step 1: Upload to GitHub

1. Go to [github.com](https://github.com) and sign in
2. Click the **+** icon → **New repository**
3. Name it `match-predictor-pro` (or any name you prefer)
4. Set to **Public** (required for free Streamlit Cloud deployment)
5. **Do NOT** initialize with README (we already have one)
6. Click **Create repository**

Then upload all files. You can either:

**Option A — Upload via GitHub web interface (easiest):**

1. On your new repo page, click **"uploading an existing file"** link
2. Drag and drop ALL files from the project folder
3. **Important:** You need to upload the folder structure correctly:
   - First upload all root files: `streamlit_app.py`, `requirements.txt`, `match_predictor_model.keras`, `streamlit_app_data.pkl`, `.gitignore`, `README.md`
   - Then create the `dataset/` folder and upload the 3 CSV files
   - Create `.streamlit/` folder and upload `config.toml`
   - Create `saved_model_match_predictor/` folder and upload its files
   - Upload the notebook file
4. Click **Commit changes**

**Option B — Upload via Git command line:**

```bash
cd /path/to/your/project/folder

git init
git add .
git commit -m "Initial commit: Match Predictor Pro"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/match-predictor-pro.git
git push -u origin main
```

> **Note:** If the `match statistics.csv` file is too large for GitHub web upload (it's ~2.9MB so should be fine), use Git command line instead.

#### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your **GitHub account**
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/match-predictor-pro`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **Deploy!**

The first deployment takes **5-10 minutes** because it installs TensorFlow. Subsequent updates deploy faster.

#### Step 3: Verify

Once deployed, you'll get a URL like:
```
https://your-app-name.streamlit.app
```

Test the full flow:
1. Log in with demo credentials
2. Select a match date
3. Choose home and away teams
4. Adjust formations and lineups
5. Click **PREDICT MATCH RESULT**

---

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/match-predictor-pro.git
cd match-predictor-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`

---

## How the Prediction Works

1. **Team Strength Calculation** — Aggregates player ratings, goals, assists, and injury data from the selected lineup
2. **Feature Engineering** — Generates a 127-dimensional feature vector from team strengths
3. **Neural Network** — Keras model (trained on 6,340 matches) outputs a base probability
4. **Schedule Adjustment** — Applies midweek fatigue and congestion penalties based on selected match date
5. **Final Probability** — Combines model output (30%) with team strength heuristics (70%) for the prediction
6. **Score Prediction** — Maps probability to realistic scorelines based on attacking strength
7. **Goal Scorer Prediction** — Weights players by position and stats to predict who scores

---

## Datasets

| Dataset | Records | Description |
|---------|---------|-------------|
| `match statistics.csv` | 6,340 | Historical EPL match results, shots, betting odds (2008-2026) |
| `epl_player_stats_24_25.csv` | 562 | Player performance stats for 2024/25 season |
| `player_injuries_impact.csv` | 656 | Injury records with before/during/after match impact |

---

## Tech Stack

- **Frontend:** Streamlit + Custom HTML/CSS (broadcast-style UI)
- **ML Model:** ONNX Runtime (converted from TensorFlow/Keras Sequential Neural Network)
- **Data:** pandas, NumPy, scikit-learn
- **Training:** Jupyter Notebook with EDA, feature engineering, and model evaluation

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Model file not found` | Ensure `match_predictor_model.onnx` is in the root directory |
| Slow first load | ONNX Runtime initializes in ~2s; much faster than TensorFlow |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Streamlit Cloud crash | App uses ONNX Runtime (not TensorFlow) — works on Python 3.11-3.14+ |
| Date picker not showing | Update Streamlit: `pip install --upgrade streamlit` |

---

## License

This project is for educational and demonstration purposes.
