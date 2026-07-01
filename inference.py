import os
import joblib
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict

# Suppress warnings for a clean terminal
warnings.filterwarnings('ignore')

from data_loader import load_tennis_dataset

def compile_live_atp_engine(raw_df):
    """
    Rapidly scans the raw timeline to construct the final, absolute present-day states
    of every player's Elo, Streaks, and EMA performance metrics.
    Includes robust filters for data anomalies and handles inactive/retired players.
    """
    print("[3/3] Compiling Live Player States & Physics Engine (With Anomaly Filters)...")

    # Sort chronologically
    df = raw_df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)

    overall_elo = defaultdict(lambda: 1500.0)
    surface_elo = defaultdict(lambda: defaultdict(lambda: 1500.0))
    streak_overall = defaultdict(int)
    streak_gslam = defaultdict(int)
    
    # NEW: Track the absolute last date a player competed to filter out retired players
    player_last_active_date = defaultdict(int)

    player_stat_history = defaultdict(lambda: {
        'svpt_won_ema': 0.62, 'revpt_won_ema': 0.38, 'ace_rate_ema': 0.05,
        'df_rate_ema': 0.03, 'first_in_ema': 0.60, 'first_won_ema': 0.70,
        'second_won_ema': 0.50, 'bp_saved_ema': 0.60, 'bp_converted_ema': 0.40
    })

    name_to_id = {}
    id_to_recent_raw = {}

    numeric_cols = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_bpSaved', 'w_bpFaced',
                    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_bpSaved', 'l_bpFaced']
    for col in numeric_cols:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else: 
            df[col] = 0.0

    for idx, row in df.iterrows():
        p1 = str(row.get('winner_id', 'Unknown')).strip()
        p2 = str(row.get('loser_id', 'Unknown')).strip()
        p1_name = row.get('winner_name_clean', 'Unknown')
        p2_name = row.get('loser_name_clean', 'Unknown')

        name_to_id[p1_name] = p1
        name_to_id[p2_name] = p2

        id_to_recent_raw[p1] = (row, 'winner')
        id_to_recent_raw[p2] = (row, 'loser')

        surface = row.get('surface', 'Hard')
        if pd.isna(surface): surface = 'Hard'
        level = row.get('tourney_level', 'A')
        if pd.isna(level): level = 'A'
        
        current_date = int(row.get('tourney_date', 19000101))
        player_last_active_date[p1] = current_date
        player_last_active_date[p2] = current_date

        # EMAs
        p1_stats = player_stat_history[p1]
        p2_stats = player_stat_history[p2]
        alpha = 0.1

        w_svpt = row['w_svpt']
        if w_svpt > 0:
            w_sv_won = row['w_1stWon'] + row['w_2ndWon']
            # PATCH: Clip values between 0.0 and 1.0 to eliminate corrupt file ratios
            sv_pct = np.clip(w_sv_won / w_svpt, 0.0, 1.0)
            p1_stats['svpt_won_ema'] = (alpha * sv_pct) + ((1 - alpha) * p1_stats['svpt_won_ema'])
            
            p1_stats['ace_rate_ema'] = (alpha * np.clip(row['w_ace'] / w_svpt, 0.0, 1.0)) + ((1 - alpha) * p1_stats['ace_rate_ema'])
            p1_stats['df_rate_ema'] = (alpha * np.clip(row['w_df'] / w_svpt, 0.0, 1.0)) + ((1 - alpha) * p1_stats['df_rate_ema'])
            p1_stats['first_in_ema'] = (alpha * np.clip(row['w_1stIn'] / w_svpt, 0.0, 1.0)) + ((1 - alpha) * p1_stats['first_in_ema'])
            if row['w_1stIn'] > 0: p1_stats['first_won_ema'] = (alpha * np.clip(row['w_1stWon'] / row['w_1stIn'], 0.0, 1.0)) + ((1 - alpha) * p1_stats['first_won_ema'])
            if (w_svpt - row['w_1stIn']) > 0: p1_stats['second_won_ema'] = (alpha * np.clip(row['w_2ndWon'] / (w_svpt - row['w_1stIn']), 0.0, 1.0)) + ((1 - alpha) * p1_stats['second_won_ema'])
            if row['w_bpFaced'] > 0: p1_stats['bp_saved_ema'] = (alpha * np.clip(row['w_bpSaved'] / row['w_bpFaced'], 0.0, 1.0)) + ((1 - alpha) * p1_stats['bp_saved_ema'])

        l_svpt = row['l_svpt']
        if l_svpt > 0:
            l_sv_won = row['l_1stWon'] + row['l_2ndWon']
            sv_pct_l = np.clip(l_sv_won / l_svpt, 0.0, 1.0)
            p2_stats['svpt_won_ema'] = (alpha * sv_pct_l) + ((1 - alpha) * p2_stats['svpt_won_ema'])
            
            p2_stats['ace_rate_ema'] = (alpha * np.clip(row['l_ace'] / l_svpt, 0.0, 1.0)) + ((1 - alpha) * p2_stats['ace_rate_ema'])
            p2_stats['df_rate_ema'] = (alpha * np.clip(row['l_df'] / l_svpt, 0.0, 1.0)) + ((1 - alpha) * p2_stats['df_rate_ema'])
            p2_stats['first_in_ema'] = (alpha * np.clip(row['l_1stIn'] / l_svpt, 0.0, 1.0)) + ((1 - alpha) * p2_stats['first_in_ema'])
            if row['l_1stIn'] > 0: p2_stats['first_won_ema'] = (alpha * np.clip(row['l_1stWon'] / row['l_1stIn'], 0.0, 1.0)) + ((1 - alpha) * p2_stats['first_won_ema'])
            if (l_svpt - row['l_1stIn']) > 0: p2_stats['second_won_ema'] = (alpha * np.clip(row['l_2ndWon'] / (l_svpt - row['l_1stIn']), 0.0, 1.0)) + ((1 - alpha) * p2_stats['second_won_ema'])
            if row['l_bpFaced'] > 0: p2_stats['bp_saved_ema'] = (alpha * np.clip(row['l_bpSaved'] / row['l_bpFaced'], 0.0, 1.0)) + ((1 - alpha) * p2_stats['bp_saved_ema'])

        if l_svpt > 0:
            l_sv_won = row['l_1stWon'] + row['l_2ndWon']
            p1_ret_won = max(0, l_svpt - l_sv_won)
            p1_stats['revpt_won_ema'] = (alpha * np.clip(p1_ret_won / l_svpt, 0.0, 1.0)) + ((1 - alpha) * p1_stats['revpt_won_ema'])
            if row['l_bpFaced'] > 0:
                p1_bp_converted = max(0, row['l_bpFaced'] - row['l_bpSaved'])
                p1_stats['bp_converted_ema'] = (alpha * np.clip(p1_bp_converted / row['l_bpFaced'], 0.0, 1.0)) + ((1 - alpha) * p1_stats['bp_converted_ema'])

        if w_svpt > 0:
            w_sv_won = row['w_1stWon'] + row['w_2ndWon']
            p2_ret_won = max(0, w_svpt - w_sv_won)
            p2_stats['revpt_won_ema'] = (alpha * np.clip(p2_ret_won / w_svpt, 0.0, 1.0)) + ((1 - alpha) * p2_stats['revpt_won_ema'])
            if row['w_bpFaced'] > 0:
                p2_bp_converted = max(0, row['w_bpFaced'] - row['w_bpSaved'])
                p2_stats['bp_converted_ema'] = (alpha * np.clip(p2_bp_converted / row['w_bpFaced'], 0.0, 1.0)) + ((1 - alpha) * p2_stats['bp_converted_ema'])

        # Streaks
        streak_overall[p1] += 1
        streak_overall[p2] = 0
        if level == 'G':
            streak_gslam[p1] += 1
            streak_gslam[p2] = 0

        # Elo
        p1_elo_pre = overall_elo[p1]
        p2_elo_pre = overall_elo[p2]
        p1_surf_elo_pre = surface_elo[p1][surface]
        p2_surf_elo_pre = surface_elo[p2][surface]

        k_factor = 32
        if row.get('dataset_type') == 'challenger': k_factor = 16
        elif level == 'G': k_factor = 40

        ea_overall = 1.0 / (1.0 + 10.0 ** ((p2_elo_pre - p1_elo_pre) / 400.0))
        overall_elo[p1] += k_factor * (1.0 - ea_overall)
        overall_elo[p2] += k_factor * (0.0 - (1.0 - ea_overall))

        ea_surf = 1.0 / (1.0 + 10.0 ** ((p2_surf_elo_pre - p1_surf_elo_pre) / 400.0))
        surface_elo[p1][surface] += k_factor * (1.0 - ea_surf)
        surface_elo[p2][surface] += k_factor * (0.0 - (1.0 - ea_surf))

    # NEW: Return the activity map alongside everything else to handle retirement filter
    return name_to_id, id_to_recent_raw, overall_elo, surface_elo, streak_overall, streak_gslam, player_stat_history, player_last_active_date

def get_player_static_data(p_id, id_to_recent_raw):
    row, outcome = id_to_recent_raw[p_id]
    prefix = 'winner_' if outcome == 'winner' else 'loser_'
    
    age = pd.to_numeric(row.get(f'{prefix}age', 0), errors='coerce')
    ht = pd.to_numeric(row.get(f'{prefix}ht', 0), errors='coerce')
    rank = pd.to_numeric(row.get(f'{prefix}rank', 999), errors='coerce')
    pts = pd.to_numeric(row.get(f'{prefix}rank_points', 0), errors='coerce')
    
    return {
        'age': age if pd.notna(age) else 0.0,
        'ht': ht if pd.notna(ht) else 180.0,
        'rank': rank if pd.notna(rank) else 999,
        'rank_points': pts if pd.notna(pts) else 0.0
    }

def calculate_h2h(raw_df, p1_name, p2_name, surface, level):
    p1_wins_all = len(raw_df[(raw_df['winner_name_clean'] == p1_name) & (raw_df['loser_name_clean'] == p2_name)])
    p2_wins_all = len(raw_df[(raw_df['winner_name_clean'] == p2_name) & (raw_df['loser_name_clean'] == p1_name)])
    
    p1_wins_surf = len(raw_df[(raw_df['winner_name_clean'] == p1_name) & (raw_df['loser_name_clean'] == p2_name) & (raw_df['surface'] == surface)])
    p2_wins_surf = len(raw_df[(raw_df['winner_name_clean'] == p2_name) & (raw_df['loser_name_clean'] == p1_name) & (raw_df['surface'] == surface)])
    
    p1_wins_lvl = len(raw_df[(raw_df['winner_name_clean'] == p1_name) & (raw_df['loser_name_clean'] == p2_name) & (raw_df['tourney_level'] == level)])
    p2_wins_lvl = len(raw_df[(raw_df['winner_name_clean'] == p2_name) & (raw_df['loser_name_clean'] == p1_name) & (raw_df['tourney_level'] == level)])

    return {
        'delta_h2h_overall': p1_wins_all - p2_wins_all,
        'delta_h2h_surface': p1_wins_surf - p2_wins_surf,
        'delta_h2h_level': p1_wins_lvl - p2_wins_lvl
    }

def launch_inference_console():
    print("="*60)
    print("🎾 ATP INFERENCE ENGINE - INITIALIZATION 🎾")
    print("="*60)
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'wimbledon_calibrated_engine.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return
    engine = joblib.load(model_path)
    print("[1/3] Calibrated XGBoost Engine Loaded.")
    
    print("[2/3] Ingesting Historical Timeline...")
    raw_df = load_tennis_dataset()
    
    # Generate live internal dictionaries
    name_map, row_map, elo_dict, surf_elo_dict, streak_dict, gs_streak_dict, stat_dict,active_dates = compile_live_atp_engine(raw_df)

    feature_columns = [
        'best_of', 'indoor', 'draw_size',
        'delta_age', 'delta_ht', 'delta_rank', 'delta_rank_points',
        'delta_elo_overall', 'delta_elo_surface',
        'delta_h2h_overall', 'delta_h2h_surface', 'delta_h2h_level',
        'delta_streak_overall', 'delta_streak_gslam',
        'delta_rest_days', 'delta_tourney_fatigue',
        'delta_serve_win_pct', 'delta_return_win_pct',
        'delta_ace_rate', 'delta_df_rate', 'delta_first_in',
        'delta_first_won', 'delta_second_won',
        'delta_bp_save_pct', 'delta_bp_conv_pct',
        'delta_dom_ratio',
        'p1_qualifier_upset_threat', 'p1_is_seeded',
        'p2_qualifier_upset_threat', 'p2_is_seeded'
    ]

    print("\n✅ SYSTEM READY. Welcome to the Wimbledon Simulator.")
    print("Type 'exit' or 'quit' at any prompt to close the engine.\n")

    while True:
        p1_name = input("Enter Player 1 Name (e.g., Carlos Alcaraz): ").strip()
        if p1_name.lower() in ['exit', 'quit']: break
            
        p2_name = input("Enter Player 2 Name (e.g., Jannik Sinner): ").strip()
        if p2_name.lower() in ['exit', 'quit']: break

        if p1_name not in name_map or p2_name not in name_map:
            print("[!] Error: One or both players not found. Please verify spelling.\n")
            continue

        try:
            surface = 'Grass'
            level = 'G'
            
            id1 = name_map[p1_name]
            id2 = name_map[p2_name]
            
            static1 = get_player_static_data(id1, row_map)
            static2 = get_player_static_data(id2, row_map)
            
            p1_serve_lost = (1.0 - stat_dict[id1]['svpt_won_ema']) + 0.001
            p1_dom = stat_dict[id1]['revpt_won_ema'] / p1_serve_lost
            
            p2_serve_lost = (1.0 - stat_dict[id2]['svpt_won_ema']) + 0.001
            p2_dom = stat_dict[id2]['revpt_won_ema'] / p2_serve_lost

            h2h_deltas = calculate_h2h(raw_df, p1_name, p2_name, surface, level)

            vector = {
                'best_of': 5, 'indoor': 0, 'draw_size': 128,
                'delta_age': static1['age'] - static2['age'],
                'delta_ht': static1['ht'] - static2['ht'],
                'delta_rank': static1['rank'] - static2['rank'],
                'delta_rank_points': static1['rank_points'] - static2['rank_points'],
                
                'delta_elo_overall': elo_dict[id1] - elo_dict[id2],
                'delta_elo_surface': surf_elo_dict[id1][surface] - surf_elo_dict[id2][surface],
                
                'delta_h2h_overall': h2h_deltas['delta_h2h_overall'],
                'delta_h2h_surface': h2h_deltas['delta_h2h_surface'],
                'delta_h2h_level': h2h_deltas['delta_h2h_level'],
                
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

            X_live = pd.DataFrame([vector])[feature_columns]
            
            prob_p1 = engine.predict_proba(X_live)[0][1] * 100
            prob_p2 = 100 - prob_p1
            
            winner = p1_name if prob_p1 > 50 else p2_name
            win_prob = max(prob_p1, prob_p2)

            print("\n" + "="*50)
            print(f"🏟️  WIMBLEDON SIMULATION: {p1_name} vs {p2_name}")
            print(f"📈 Overall Elo: {p1_name} ({elo_dict[id1]:.0f}) | {p2_name} ({elo_dict[id2]:.0f})")
            print(f"🌱 Grass Elo:   {p1_name} ({surf_elo_dict[id1][surface]:.0f}) | {p2_name} ({surf_elo_dict[id2][surface]:.0f})")
            print("-"*50)
            print(f"🏆 PREDICTED WINNER: {winner.upper()}")
            print(f"🎯 WIN PROBABILITY:  {win_prob:.1f}%")
            print("="*50 + "\n")

        except Exception as e:
            print(f"\n[!] Inference Error: {e}\n")

if __name__ == "__main__":
    launch_inference_console()