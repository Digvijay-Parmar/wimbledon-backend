import pandas as pd
import glob
import os

def load_tennis_dataset(base_dir=None):
    if base_dir is None:
        # This points to a 'tml-data' folder in the same directory as the script
        base_dir = os.path.join(os.path.dirname(__file__), 'tml-data')
    
    # 1. Gather all Main Tour CSV files
    all_csvs = glob.glob(os.path.join(base_dir, "[0-9][0-9][0-9][0-9].csv"))
    main_list = []
    for f in all_csvs:
        df = pd.read_csv(f)
        df['dataset_type'] = 'main'
        main_list.append(df)
        
    if not main_list:
        raise FileNotFoundError(f"Error: No annual main tour CSV files found in {base_dir}")
    main_df = pd.concat(main_list, ignore_index=True)
    print(f"Loaded {len(main_df)} Main Tour matches.")

    # 2. Gather all Challenger CSV files
    challenger_files = glob.glob(os.path.join(base_dir, "*_challenger.csv"))
    chal_list = []
    for f in challenger_files:
        df = pd.read_csv(f)
        df['dataset_type'] = 'challenger'
        chal_list.append(df)
        
    if chal_list:
        chal_df = pd.concat(chal_list, ignore_index=True)
        print(f"Loaded {len(chal_df)} Challenger matches.")
    else:
        print("Warning: No Challenger files found.")
        chal_df = pd.DataFrame()

    # 3. Gather all Qualification CSV files
    quali_dir = os.path.join(base_dir, "atp_quali")
    quali_files = glob.glob(os.path.join(quali_dir, "*_atp_quali.csv"))
    
    if not quali_files:
        quali_files = glob.glob(os.path.join(quali_dir, "*.csv"))
        
    qua_list = []
    for f in quali_files:
        df = pd.read_csv(f)
        df['dataset_type'] = 'quali'
        if 'winner_id' in df.columns and 'loser_id' in df.columns:
            qua_list.append(df)
            
    if qua_list:
        qua_df = pd.concat(qua_list, ignore_index=True)
        print(f"Loaded {len(qua_df)} Qualification matches.")
    else:
        print("Warning: No Qualification files found.")
        qua_df = pd.DataFrame()

    # 4. Master Chronological Merger
    all_matrices = [main_df, chal_df, qua_df]
    combined_df = pd.concat([m for m in all_matrices if not m.empty], ignore_index=True)
    
    combined_df['tourney_date'] = pd.to_numeric(combined_df['tourney_date'], errors='coerce')
    combined_df = combined_df.dropna(subset=['tourney_date', 'winner_id', 'loser_id'])
    combined_df = combined_df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)

    # 5. Load Player Database Metadata (Safely forcing string type to prevent .str errors)
    player_file = os.path.join(base_dir, "ATP_database.csv")
    if os.path.exists(player_file):
        print(f"Loading player database metadata from: {player_file}")
        # FIX: dtype=str forces Pandas to read everything as text first
        players_df = pd.read_csv(player_file, dtype=str) 
        
        players_df.columns = players_df.columns.str.replace('"', '').str.strip()
        
        id_col = 'id' if 'id' in players_df.columns else players_df.columns[0]
        players_df[id_col] = players_df[id_col].str.replace('"', '').str.strip()
        
        name_col = 'atpname' if 'atpname' in players_df.columns else ('player' if 'player' in players_df.columns else None)
        if name_col:
            players_df[name_col] = players_df[name_col].fillna('').str.replace('"', '').str.strip()
            name_map = dict(zip(players_df[id_col], players_df[name_col]))
            combined_df['winner_name_clean'] = combined_df['winner_id'].astype(str).map(name_map).fillna(combined_df['winner_name'])
            combined_df['loser_name_clean'] = combined_df['loser_id'].astype(str).map(name_map).fillna(combined_df['loser_name'])
            
        hand_col = 'hand' if 'hand' in players_df.columns else None
        if hand_col:
            players_df[hand_col] = players_df[hand_col].fillna('').str.replace('"', '').str.strip()
            hand_map = dict(zip(players_df[id_col], players_df[hand_col]))
            combined_df['winner_hand'] = combined_df['winner_id'].astype(str).map(hand_map).fillna(combined_df['winner_hand'])
            combined_df['loser_hand'] = combined_df['loser_id'].astype(str).map(hand_map).fillna(combined_df['loser_hand'])
            
        ht_col = 'height' if 'height' in players_df.columns else None
        if ht_col:
            players_df[ht_col] = pd.to_numeric(players_df[ht_col].fillna('').str.replace('"', '').str.strip(), errors='coerce')
            ht_map = dict(zip(players_df[id_col], players_df[ht_col]))
            combined_df['winner_ht'] = combined_df['winner_id'].astype(str).map(ht_map).fillna(combined_df['winner_ht'])
            combined_df['loser_ht'] = combined_df['loser_id'].astype(str).map(ht_map).fillna(combined_df['loser_ht'])
    else:
        print("Warning: ATP_database.csv not found. Relying on default file attributes.")
        combined_df['winner_name_clean'] = combined_df['winner_name']
        combined_df['loser_name_clean'] = combined_df['loser_name']

    # 6. Fill common missing structural values safely
    combined_df['w_ace'] = pd.to_numeric(combined_df['w_ace'], errors='coerce').fillna(0)
    combined_df['l_ace'] = pd.to_numeric(combined_df['l_ace'], errors='coerce').fillna(0)
    combined_df['draw_size'] = pd.to_numeric(combined_df['draw_size'], errors='coerce').fillna(128)
    
    print(f"Data Loader verification complete. Unified Timeline contains {len(combined_df)} match vectors.")
    return combined_df

if __name__ == "__main__":
    try:
        df = load_tennis_dataset()
        print("\n>>> Data Loader verification successful! Ready for feature engineering. <<<")
    except Exception as e:
        print(f"\n>>> Data Loader verification failed: {e} <<<")