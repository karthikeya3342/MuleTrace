import pandas as pd
import numpy as np
import json
import os

dataset_path = "DataSet.csv"
if not os.path.exists(dataset_path):
    print("Error: DataSet.csv not found")
    exit(1)

df = pd.read_csv(dataset_path)

# Load hybrid features
with open("hybrid_features.json", "r") as f:
    hybrid_info = json.load(f)
    hybrid_features = hybrid_info["hybrid_features"]

# Keep index, target and hybrid features
cols_to_keep = ['Unnamed: 0', 'F3924'] + hybrid_features
df_sub = df[cols_to_keep].copy()

# Rename Unnamed: 0 to AccountId
df_sub = df_sub.rename(columns={'Unnamed: 0': 'account_id'})

# Separate clean and mule accounts
clean_accounts = df_sub[df_sub['F3924'] == 0]
mule_accounts = df_sub[df_sub['F3924'] == 1]

# Sample 35 clean accounts and all 15 available if we want a mix (we have 81 mules)
# Let's take 35 clean and 15 mules
sampled_clean = clean_accounts.sample(n=35, random_state=42)
sampled_mules = mule_accounts.sample(n=15, random_state=42)

sampled_all = pd.concat([sampled_clean, sampled_mules]).sample(frac=1.0, random_state=42) # Shuffle

# Fill NaNs with string "NaN" or None so it is valid JSON
sampled_all = sampled_all.replace({np.nan: None})

# Convert to list of dicts
accounts_list = []
for idx, row in sampled_all.iterrows():
    acc_dict = row.to_dict()
    # Convert types to standard Python types for JSON
    acc_dict['account_id'] = f"ACC{int(acc_dict['account_id']):05d}"
    acc_dict['is_mule'] = int(acc_dict['F3924'])
    del acc_dict['F3924']
    
    # Clean up other numeric types
    for k, v in acc_dict.items():
        if isinstance(v, (np.integer, np.int64)):
            acc_dict[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            acc_dict[k] = float(v)
            
    accounts_list.append(acc_dict)

with open("sample_accounts.json", "w") as f:
    json.dump(accounts_list, f, indent=4)

print(f"Successfully saved {len(accounts_list)} sample accounts to 'sample_accounts.json'.")
