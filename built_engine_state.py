import os
import joblib
import pandas as pd
from collections import defaultdict
from data_loader import load_tennis_dataset
from inference import compile_live_atp_engine

def build_and_save_state():
    print("Loading raw datasets...")
    raw_df = load_tennis_dataset()
    
    print("Compiling live player states...")
    name_map, row_map, elo_dict, surf_elo_dict, streak_dict, gs_streak_dict, stat_dict, active_dates = compile_live_atp_engine(raw_df)
    
    print("Pre-calculating static player profiles...")
    # Instead of keeping full pandas rows in memory, we extract only what we need
    player_static_profiles = {}
    for p_id, (row, outcome) in row_map.items():
        prefix = 'winner_' if outcome == 'winner' else 'loser_'
        
        age = pd.to_numeric(row.get(f'{prefix}age', 0), errors='coerce')
        ht = pd.to_numeric(row.get(f'{prefix}ht', 0), errors='coerce')
        rank = pd.to_numeric(row.get(f'{prefix}rank', 999), errors='coerce')
        pts = pd.to_numeric(row.get(f'{prefix}rank_points', 0), errors='coerce')
        
        player_static_profiles[p_id] = {
            'age': float(age) if pd.notna(age) else 0.0,
            'ht': float(ht) if pd.notna(ht) else 180.0,
            'rank': float(rank) if pd.notna(rank) else 999.0,
            'rank_points': float(pts) if pd.notna(pts) else 0.0
        }

    print("Pre-calculating global Head-to-Head matrix...")
    # This completely eliminates the need to hold raw_df in RAM
    h2h_matrix = defaultdict(lambda: defaultdict(int))
    for _, row in raw_df.iterrows():
        w_name = row.get('winner_name_clean')
        l_name = row.get('loser_name_clean')
        if pd.isna(w_name) or pd.isna(l_name):
            continue
            
        surface = row.get('surface', 'Hard')
        level = row.get('tourney_level', 'A')
        
        h2h_matrix[(w_name, l_name)]['overall'] += 1
        h2h_matrix[(w_name, l_name)][surface] += 1
        h2h_matrix[(w_name, l_name)][level] += 1

    # Package everything into a lightweight dictionary
    system_state = {
        'name_map': name_map,
        'elo_dict': dict(elo_dict),
        'surf_elo_dict': {k: dict(v) for k, v in surf_elo_dict.items()},
        'streak_dict': dict(streak_dict),
        'gs_streak_dict': dict(gs_streak_dict),
        'stat_dict': {k: dict(v) for k, v in stat_dict.items()},
        'player_static_profiles': player_static_profiles,
        'h2h_matrix': dict(h2h_matrix)
    }

    # Save to the models directory
    save_path = os.path.join(os.path.dirname(__file__), 'models', 'engine_state.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(system_state, save_path)
    print(f"✅ State successfully compressed and saved to {save_path}!")

if __name__ == "__main__":
    build_and_save_state()