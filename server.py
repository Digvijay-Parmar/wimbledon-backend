from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'matches.db')
db = SQLAlchemy(app)

class MatchResult(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    winner_id = db.Column(db.Integer, nullable=True)
    ibm_prob1 = db.Column(db.Float, nullable=True)

with app.app_context():
    db.create_all()

# --- Engine Setup (Optimized for Low RAM) ---
print("--- Loading Pre-computed Engine State ---")
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# 1. Load the ML Model
model = joblib.load(os.path.join(models_dir, 'wimbledon_calibrated_engine.pkl'))

# 2. Load the pre-computed dictionary state (No Pandas parsing needed)
state = joblib.load(os.path.join(models_dir, 'engine_state.pkl'))
name_map = state['name_map']
elo_dict = state['elo_dict']
surf_elo_dict = state['surf_elo_dict']
streak_dict = state['streak_dict']
gs_streak_dict = state['gs_streak_dict']
stat_dict = state['stat_dict']
player_static = state['player_static_profiles']
h2h_matrix = state['h2h_matrix']
print("--- Engine Ready ---")

# Helpers for the new architecture
def get_h2h_delta(p1, p2, surface, level):
    p1_wins_all = h2h_matrix.get((p1, p2), {}).get('overall', 0)
    p2_wins_all = h2h_matrix.get((p2, p1), {}).get('overall', 0)
    
    p1_wins_surf = h2h_matrix.get((p1, p2), {}).get(surface, 0)
    p2_wins_surf = h2h_matrix.get((p2, p1), {}).get(surface, 0)
    
    p1_wins_lvl = h2h_matrix.get((p1, p2), {}).get(level, 0)
    p2_wins_lvl = h2h_matrix.get((p2, p1), {}).get(level, 0)
    
    return {
        'delta_h2h_overall': p1_wins_all - p2_wins_all,
        'delta_h2h_surface': p1_wins_surf - p2_wins_surf,
        'delta_h2h_level': p1_wins_lvl - p2_wins_lvl
    }

@app.route('/api/get-all-matches', methods=['GET'])
def get_all_matches():
    results = MatchResult.query.all()
    return jsonify({res.id: {"winner_id": res.winner_id, "ibm_prob1": res.ibm_prob1} for res in results})

@app.route('/api/update-match', methods=['POST'])
def update_match():
    data = request.json
    match_id = data['id']
    match = MatchResult.query.get(match_id) or MatchResult(id=match_id)
    if 'winner_id' in data: match.winner_id = data['winner_id']
    if 'ibm_prob1' in data:
        match.ibm_prob1 = float(data['ibm_prob1']) if data['ibm_prob1'] is not None else None
    db.session.add(match)
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    p1_name = data.get('p1')
    p2_name = data.get('p2')

    try:
        if p1_name not in name_map or p2_name not in name_map:
            raise ValueError(f"Player not found: {p1_name} or {p2_name}")

        id1 = name_map[p1_name]
        id2 = name_map[p2_name]

        static1 = player_static[id1]
        static2 = player_static[id2]

        p1_serve_lost = (1.0 - stat_dict[id1]['svpt_won_ema']) + 0.001
        p1_dom = stat_dict[id1]['revpt_won_ema'] / p1_serve_lost
        p2_serve_lost = (1.0 - stat_dict[id2]['svpt_won_ema']) + 0.001
        p2_dom = stat_dict[id2]['revpt_won_ema'] / p2_serve_lost

        h2h = get_h2h_delta(p1_name, p2_name, 'Grass', 'G')

        vector = {
            'best_of': 5, 'indoor': 0, 'draw_size': 128,
            'delta_age': static1['age'] - static2['age'],
            'delta_ht': static1['ht'] - static2['ht'],
            'delta_rank': static1['rank'] - static2['rank'],
            'delta_rank_points': static1['rank_points'] - static2['rank_points'],
            'delta_elo_overall': elo_dict[id1] - elo_dict[id2],
            'delta_elo_surface': surf_elo_dict[id1].get('Grass', 1500.0) - surf_elo_dict[id2].get('Grass', 1500.0),
            'delta_h2h_overall': h2h['delta_h2h_overall'],
            'delta_h2h_surface': h2h['delta_h2h_surface'],
            'delta_h2h_level': h2h['delta_h2h_level'],
            'delta_streak_overall': streak_dict[id1] - streak_dict[id2],
            'delta_streak_gslam': gs_streak_dict[id1] - gs_streak_dict[id2],
            'delta_rest_days': 0, 'delta_tourney_fatigue': 0,
            'delta_serve_win_pct': stat_dict[id1]['svpt_won_ema'] - stat_dict[id2]['svpt_won_ema'],
            'delta_return_win_pct': stat_dict[id1]['revpt_won_ema'] - stat_dict[id2]['revpt_won_ema'],
            'delta_ace_rate': stat_dict[id1]['ace_rate_ema'] - stat_dict[id2]['ace_rate_ema'],
            'delta_df_rate': stat_dict[id1]['df_rate_ema'] - stat_dict[id2]['df_rate_ema'],
            'delta_first_in': stat_dict[id1]['first_in_ema'] - stat_dict[id2]['first_in_ema'],
            'delta_first_won': stat_dict[id1]['first_won_ema'] - stat_dict[id2]['first_won_ema'],
            'delta_second_won': stat_dict[id1]['second_won_ema'] - stat_dict[id2]['second_won_ema'],
            'delta_bp_save_pct': stat_dict[id1]['bp_saved_ema'] - stat_dict[id2]['bp_saved_ema'],
            'delta_bp_conv_pct': stat_dict[id1]['bp_converted_ema'] - stat_dict[id2]['bp_converted_ema'],
            'delta_dom_ratio': p1_dom - p2_dom,
            'p1_qualifier_upset_threat': 0, 'p1_is_seeded': 1 if static1['rank'] <= 32 else 0,
            'p2_qualifier_upset_threat': 0, 'p2_is_seeded': 1 if static2['rank'] <= 32 else 0,
        }

        feature_order = [
            'best_of', 'indoor', 'draw_size', 'delta_age', 'delta_ht', 'delta_rank', 'delta_rank_points',
            'delta_elo_overall', 'delta_elo_surface', 'delta_h2h_overall', 'delta_h2h_surface', 'delta_h2h_level',
            'delta_streak_overall', 'delta_streak_gslam', 'delta_rest_days', 'delta_tourney_fatigue',
            'delta_serve_win_pct', 'delta_return_win_pct', 'delta_ace_rate', 'delta_df_rate', 'delta_first_in',
            'delta_first_won', 'delta_second_won', 'delta_bp_save_pct', 'delta_bp_conv_pct', 'delta_dom_ratio',
            'p1_qualifier_upset_threat', 'p1_is_seeded', 'p2_qualifier_upset_threat', 'p2_is_seeded'
        ]

        df_pred = pd.DataFrame([vector])[feature_order]
        prob_p1 = model.predict_proba(df_pred)[0][1] * 100

        return jsonify({
            "p1_prob": round(float(prob_p1), 1),
            "p2_prob": round(100 - float(prob_p1), 1)
        })

    except Exception as e:
        print(f"CRITICAL ERROR for {p1_name} vs {p2_name}: {e}")
        return jsonify({"p1_prob": None, "p2_prob": None})

if __name__ == '__main__':
    app.run(port=5000, debug=False)