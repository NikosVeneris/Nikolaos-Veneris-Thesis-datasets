import pandas as pd
import numpy as np
import itertools
import math

# 1. Base Attributes 
COST_LEVELS = [4,6,8]
NOTIF_LEVELS = [1, 2, 3, 4, 5] # 1=Home, 2=Stop, 3=AccSAV, 4=Train, 5=EgrWait
IVT_ACCESS_LEVELS = [4, 6, 8]
IVT_EGRESS_LEVELS = [4, 6, 8]
BUFFER_TIME_LEVELS = [-5, 0, 5] # NEW: Added buffer time levels 

# 2. Independent Variance Sources (Sigmas - Reduced to 4)
SIGMA_WAIT_ACCESS_LEVELS = [0, 2, 4] # Merged Home & Stop
SIGMA_WAIT_EGRESS_LEVELS = [0, 2, 4]
SIGMA_IVT_ACCESS_LEVELS = [0, 2, 4]
SIGMA_IVT_EGRESS_LEVELS = [0, 2, 4]

# 3. Categorical Delay Reasons (1=Traffic, 2=Reroute, 3=software_issue)
PROFILE_LEVELS = [1, 2, 3]

# 4. Discrete Lognormal Profiles (Quintiles: 10%, 30%, 50%, 70%, 90%)
WAIT_PROFILES = {
    0: [0, 0, 0, 0, 0],
    2: [1, 2, 3, 4, 6],   
    4: [0, 1, 3, 5, 10]   
}

IVT_PROFILES = {
    0: [0, 0, 0, 0, 0],
    2: [1, 2, 3, 4, 6],  
    4: [0, 1, 3, 5, 10]  
}

# Constants
WALK_TIME = 3
TRAIN_IVT = 6
TRAIN_FREQ = 7
TARGET_WORK_START = 660 # Absolute minute anchor (e.g., 11:00 AM) to lock UI timelines

print("Generating full factorial...")
all_combinations = list(itertools.product(
    COST_LEVELS, NOTIF_LEVELS, IVT_ACCESS_LEVELS, IVT_EGRESS_LEVELS,
    SIGMA_WAIT_ACCESS_LEVELS, # <-- Merged sigma
    SIGMA_WAIT_EGRESS_LEVELS, SIGMA_IVT_ACCESS_LEVELS, SIGMA_IVT_EGRESS_LEVELS,
    PROFILE_LEVELS, BUFFER_TIME_LEVELS
))

valid_profiles = []

for combo in all_combinations:
    cost, notif, ivt_a, ivt_e, sig_w_a, sig_w_e, sig_i_a, sig_i_e, prof, buffer = combo
    
    # CONSTRAINT 1: Strictly ONE varying travel time per profile
    sources = sum([sig_w_a > 0, sig_w_e > 0, sig_i_a > 0, sig_i_e > 0])
    if sources > 1:
        continue

    # NEW: is_reliable dummy
    is_reliable = 1 if sources == 0 else 0

    # Fallback logic ONLY for reliable journeys
    if sources == 0:
        if prof != 1: continue 
        if notif != 1: continue # Prevent duplicating reliable profiles across notif levels
        prof = 0  
        notif = 0

    # CONSTRAINT 2: Chronological Notification Logic
    # 1. Access Pickup Delay -> Must be notified at home (1) or at stop (2)
    if sig_w_a > 0 and notif > 2: 
        continue 
    # 2. Access IVT delay -> Must be notified before getting out of Access SAV (3)
    if sig_i_a > 0 and notif > 3: 
        continue
    # 3. Egress delays -> Must be notified before entering the egress SAV 
    if sig_w_e > 0 and notif > 4: 
        continue

    # Text descriptor for Qualtrics: Reason for delay
    if prof == 1: reason_text = "Traffic"
    elif prof == 2: reason_text = "Passenger Pickup"
    elif prof == 3: reason_text = "Software issue"
    else: reason_text = "None"

    # Text descriptor for Qualtrics: Delay
    delay_text = "No delay"
    if sig_w_a > 0 and prof == 1: delay_text = "Delay in first SAV pickup due to traffic"
    elif sig_w_a > 0 and prof == 2: delay_text = "Delay in first SAV pickup, as it picked up more passengers before you"
    elif sig_w_a > 0 and prof == 3: delay_text = "Delay in first SAV pickup, as it had to make a software update"
    elif sig_i_a > 0 and prof == 1: delay_text = "Delay in first SAV ride due to traffic"
    elif sig_i_a > 0 and prof == 2: delay_text = "Delay in first SAV ride, as it picked up more passengers along the way"
    elif sig_i_a > 0 and prof == 3: delay_text = "Delay in first SAV ride, as it had to make a software update"
    elif sig_w_e > 0 and prof == 1: delay_text = "Delay in second SAV pickup due to traffic"
    elif sig_w_e > 0 and prof == 2: delay_text = "Delay in second SAV pickup, as it picked up more passengers before you"
    elif sig_w_e > 0 and prof == 3: delay_text = "Delay in second SAV pickup, as it had to make a software update"
    elif sig_i_e > 0 and prof == 1: delay_text = "Delay in second SAV ride due to traffic"
    elif sig_i_e > 0 and prof == 2: delay_text = "Delay in second SAV ride, as it picked up more passengers along the way"
    elif sig_i_e > 0 and prof == 3: delay_text = "Delay in second SAV ride, as it had to make a software update"

    # Text descriptor for Qualtrics: Notification
    notif_text = "N/A"
    if notif == 1: notif_text = "At home"
    elif notif == 2: notif_text = "At the first pickup point"
    elif notif == 3: notif_text = "In first SAV"
    elif notif == 4: notif_text = "In train"
    elif notif == 5: notif_text = "In second SAV" # <-- New 5th Level
        
    # Unified Dummy coding for Ngene
    notif_home = 1 if notif == 1 else 0
    notif_stop = 1 if notif == 2 else 0
    notif_acc_sav = 1 if notif == 3 else 0
    notif_train = 1 if notif == 4 else 0
    notif_egr_sav = 1 if notif == 5 else 0

    arr_w_a = WAIT_PROFILES[sig_w_a]
    arr_w_e = WAIT_PROFILES[sig_w_e]
    arr_i_a = IVT_PROFILES[sig_i_a]
    arr_i_e = IVT_PROFILES[sig_i_e]
    
    # Dynamic Target Arrival based on current buffer level
    target_arrival = TARGET_WORK_START - buffer
    
    # ABSOLUTE TIMELINE ANCHORING
    perfect_train_dep = math.floor((target_arrival - TRAIN_IVT - ivt_e) / TRAIN_FREQ) * TRAIN_FREQ
    agreed_pickup = perfect_train_dep - ivt_a
    
    bars_w_h, bars_w_s, bars_w_t, bars_w_e = [], [], [], []
    bars_i_a, bars_i_e = [], []
    sde_bars, sdl_bars = [], []
    
    train_deps = [] 
    
    for k in range(5):
        # 1. Access Leg 
        pickup_delay = arr_w_a[k] 
        actual_pickup = agreed_pickup + pickup_delay
        
        # Commuter adjusts leave_home time ONLY if notified "At home" (notif == 1)
        if notif == 1:
            leave_home = actual_pickup - WALK_TIME
        else:
            leave_home = agreed_pickup - WALK_TIME
            
        journey_start = agreed_pickup - WALK_TIME
        w_h = leave_home - journey_start
        
        arrive_at_stop = leave_home + WALK_TIME
        w_s = max(0, actual_pickup - arrive_at_stop)
        
        act_i_a = ivt_a + arr_i_a[k]
        t_station = actual_pickup + act_i_a
        
        # 2. Train Connection 
        actual_train_dep = math.ceil(t_station / TRAIN_FREQ) * TRAIN_FREQ
        w_t = actual_train_dep - t_station
        
        train_deps.append(actual_train_dep)
        
        # 3. Egress Leg
        t_station_dest = actual_train_dep + TRAIN_IVT
        w_e = arr_w_e[k]
        act_i_e = ivt_e + arr_i_e[k]
        
        t_arrival = t_station_dest + w_e + act_i_e
        
        bars_w_h.append(w_h)
        bars_w_s.append(w_s)
        bars_w_t.append(w_t)
        bars_i_a.append(act_i_a)
        bars_w_e.append(w_e)
        bars_i_e.append(act_i_e)
        
        sde_bars.append(max(0, TARGET_WORK_START - t_arrival))
        sdl_bars.append(max(0, t_arrival - TARGET_WORK_START))

    earliest_train = min(train_deps)
    count_earliest = train_deps.count(earliest_train)
    prob_earliest_train = count_earliest / 5.0

    # Base Dummies
    dummy_traffic = 1 if prof == 1 else 0
    dummy_reroute = 1 if prof == 2 else 0
    dummy_software_issue = 1 if prof == 3 else 0

    profile_dict = {
        'is_reliable': is_reliable,
        'cost': cost, 'notif': notif, 'delay_text': delay_text, 'reason_code': prof, 
        'notif_text': notif_text,  
        'buffer_time': buffer,  
        
        'notif_home': notif_home, 'notif_stop': notif_stop,
        'notif_acc_sav': notif_acc_sav, 'notif_train': notif_train, 'notif_egr_sav': notif_egr_sav,
        
        'prob_earliest_train': prob_earliest_train,
        
        'exp_wait_home': np.mean(bars_w_h), 
        'exp_wait_stop': np.mean(bars_w_s), 
        'exp_wait_train': np.mean(bars_w_t), 
        'exp_wait_egress': np.mean(bars_w_e),
        'exp_ivt_access': np.mean(bars_i_a), 
        'exp_ivt_egress': np.mean(bars_i_e),
        'exp_sde': np.mean(sde_bars), 
        'exp_sdl': np.mean(sdl_bars),
        
        # Ngene Variance Identifiers
        'sigma_wait_access': sig_w_a, # <-- Merged Sigma
        'sigma_wait_egress': sig_w_e,
        'sigma_ivt_access': sig_i_a, 'sigma_ivt_egress': sig_i_e,
        
        'dummy_traffic': dummy_traffic, 
        'dummy_reroute': dummy_reroute, 
        'dummy_software_issue': dummy_software_issue
    }
    
    for i in range(5):
        profile_dict[f'wait_home_bar_{i+1}'] = bars_w_h[i]
        profile_dict[f'wait_stop_bar_{i+1}'] = bars_w_s[i]
        profile_dict[f'wait_train_bar_{i+1}'] = bars_w_t[i]
        profile_dict[f'wait_egr_bar_{i+1}'] = bars_w_e[i]
        profile_dict[f'ivt_acc_bar_{i+1}'] = bars_i_a[i]
        profile_dict[f'ivt_egr_bar_{i+1}'] = bars_i_e[i]
        profile_dict[f'sde_bar_{i+1}'] = sde_bars[i]
        profile_dict[f'sdl_bar_{i+1}'] = sdl_bars[i]

    valid_profiles.append(profile_dict)

df = pd.DataFrame(valid_profiles)
print(f"Generated {len(df)} valid discrete profiles.")

np.random.seed(42)
n_pairs = 10000
idx1 = np.random.randint(0, len(df), n_pairs)
idx2 = np.random.randint(0, len(df), n_pairs)

for i in range(n_pairs):
    while idx1[i] == idx2[i]: idx2[i] = np.random.randint(0, len(df))

df_alt1 = df.iloc[idx1].reset_index(drop=True).add_prefix("alt1.")
df_alt2 = df.iloc[idx2].reset_index(drop=True).add_prefix("alt2.")

# --- EXPORT 1: Base Excel File ---
df_base = pd.concat([df_alt1, df_alt2], axis=1)
df_base.insert(0, 'Choice situation', range(1, n_pairs + 1))

# --- EXPORT 2: Strict Ngene Excel ---
cols_m1 = [
    'is_reliable', 'cost', 'prob_earliest_train',
    'dummy_reroute', 'dummy_software_issue',
    'notif_stop', 'notif_acc_sav', 'notif_train', 'notif_egr_sav',
    'exp_wait_home', 'exp_wait_stop', 'exp_wait_train', 'exp_wait_egress', 'exp_ivt_access', 'exp_ivt_egress', 
    'exp_sde', 'exp_sdl'
]

cols_m2 = [
    'is_reliable', 'cost', 
    'dummy_reroute', 'dummy_software_issue',
    'notif_stop', 'notif_acc_sav', 'notif_train', 'notif_egr_sav',
    'exp_wait_train', 'exp_ivt_access', 'exp_ivt_egress', 
    'sigma_wait_access', 'sigma_wait_egress', 'sigma_ivt_access', 'sigma_ivt_egress'
]

df1_m1 = df_alt1[[f"alt1.{c}" for c in cols_m1]].copy()
df2_m1 = df_alt2[[f"alt2.{c}" for c in cols_m1]].copy()
df1_m2 = df_alt1[[f"alt1.{c}" for c in cols_m2]].copy()
df2_m2 = df_alt2[[f"alt2.{c}" for c in cols_m2]].copy()

df_ngene = pd.concat([df1_m1, df2_m1, df1_m2, df2_m2], axis=1)
df_ngene.insert(0, 'Choice situation', range(1, n_pairs + 1))
df_ngene.to_excel("cs_ngene_strict_scenario_1.xlsx", index=False)
print("Saved Export 2: cs_ngene_strict_scenario_1.xlsx ")

# --- EXPORT 3: Qualtrics Rich Excel ---
df_base.to_excel("cs_qualtrics_rich_scenario_1.xlsx", index=False)
print("Saved Export 3: cs_qualtrics_rich_scenario_1.xlsx")