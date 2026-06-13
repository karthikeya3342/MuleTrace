from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import json
import os
import random

# Initialize FastAPI app
app = FastAPI(
    title="MuleTrace (MTX) AML API",
    description="Real-Time AI/ML-Powered Mule Account Classification and Fraud Detection API",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# File Paths
METADATA_PATH = "model_metadata.json"
XGB_MODEL_PATH = "xgb_hybrid_model.json"
LGB_MODEL_PATH = "lgb_hybrid_model.txt"
SAMPLE_ACCOUNTS_PATH = "sample_accounts.json"

# Global Variables to hold models and metadata
metadata = None
xgb_model = None
lgb_model = None
sample_accounts = []
hybrid_features = []
medians = {}
categorical_mappings = {}

@app.on_event("startup")
def load_resources():
    global metadata, xgb_model, lgb_model, sample_accounts, hybrid_features, medians, categorical_mappings
    
    print("Initializing backend resources...")
    
    # 1. Load Metadata
    if not os.path.exists(METADATA_PATH):
        raise RuntimeError(f"Metadata file '{METADATA_PATH}' not found. Run 'train_and_serialize.py' first.")
    
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    
    hybrid_features = metadata["hybrid_features"]
    medians = metadata["medians"]
    categorical_mappings = metadata["categorical_mappings"]
    print(f"Loaded metadata. {len(hybrid_features)} features configured.")
    
    # 2. Load XGBoost Model
    if not os.path.exists(XGB_MODEL_PATH):
        raise RuntimeError(f"XGBoost model file '{XGB_MODEL_PATH}' not found.")
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    print("XGBoost model loaded successfully.")
    
    # 3. Load LightGBM Model
    if not os.path.exists(LGB_MODEL_PATH):
        raise RuntimeError(f"LightGBM model file '{LGB_MODEL_PATH}' not found.")
    
    lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)
    print("LightGBM model loaded successfully.")
    
    # 4. Load Sample Accounts
    if os.path.exists(SAMPLE_ACCOUNTS_PATH):
        with open(SAMPLE_ACCOUNTS_PATH, "r") as f:
            sample_accounts = json.load(f)
        print(f"Loaded {len(sample_accounts)} sample accounts.")
        
        # Inject deterministic device_fingerprint for 3D entity resolution
        import hashlib
        for acc in sample_accounts:
            acc_id = str(acc.get("account_id", ""))
            branch = acc.get("F3887")
            occupation = acc.get("F3891")
            branch_str = str(branch) if branch is not None else "UNKNOWN"
            occ_str = str(occupation).lower().strip() if occupation is not None else "unknown"
            
            # Create a deterministic sub-cluster index (0, 1, or 2) using md5 of account_id
            h = hashlib.md5(acc_id.encode("utf-8")).hexdigest()
            cluster_idx = int(h, 16) % 3
            
            # Combine to form the device fingerprint hash
            fp_raw = f"BR-{branch_str}_OCC-{occ_str}_CL-{cluster_idx}"
            fp_hash = hashlib.md5(fp_raw.encode("utf-8")).hexdigest()[:8].upper()
            acc["device_fingerprint"] = f"DEV-{fp_hash}"
    else:
        print("Warning: sample_accounts.json not found. Simulation features will be limited.")

# Pydantic schema for prediction request
class PredictRequest(BaseModel):
    features: dict
    base_features: dict = None
    engine: str = "xgboost"  # "xgboost" or "lightgbm"
    threshold: float = 0.5

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/status")
def read_status():
    return {
        "status": "online",
        "service": "MuleTrace (MTX) AML API",
        "models_configured": ["xgboost", "lightgbm"],
        "total_features": len(hybrid_features)
    }

@app.get("/metrics")
def get_metrics():
    if not metadata:
        raise HTTPException(status_code=500, detail="Metadata not loaded.")
    return metadata["metrics"]

@app.get("/metadata")
def get_metadata():
    if not metadata:
        raise HTTPException(status_code=500, detail="Metadata not loaded.")
    return {
        "hybrid_features": metadata["hybrid_features"],
        "medians": metadata["medians"],
        "categorical_mappings": metadata["categorical_mappings"],
        "cohort_medians": metadata.get("cohort_medians", {})
    }

@app.get("/accounts")
def get_accounts():
    return sample_accounts

@app.get("/simulate")
def simulate_incoming_transaction():
    if not sample_accounts:
        raise HTTPException(status_code=500, detail="Sample accounts not loaded.")
    
    # Select a random account
    account = random.choice(sample_accounts)
    
    # Create a copy to perturb slightly and make it unique
    simulated = account.copy()
    
    # Generate a new random Account ID for the stream
    rand_num = random.randint(10000, 99999)
    simulated["account_id"] = f"ACC{rand_num:05d}"
    
    # Optionally perturb some numeric features slightly to simulate real variations
    for col in hybrid_features:
        # Avoid perturbing categorical columns (which are in categorical_mappings)
        if col not in categorical_mappings and simulated[col] is not None:
            # Add small noise (up to 5%)
            val = simulated[col]
            if isinstance(val, (int, float)) and val != 0:
                noise = random.uniform(-0.05, 0.05) * val
                simulated[col] = type(val)(val + noise)
                
    return simulated

@app.post("/predict")
def predict_account(request: PredictRequest):
    global xgb_model, lgb_model, hybrid_features, medians, categorical_mappings
    
    # 1. Validate engine
    engine = request.engine.lower().strip()
    if engine not in ["xgboost", "lightgbm"]:
        raise HTTPException(status_code=400, detail="Invalid engine. Must be 'xgboost' or 'lightgbm'.")
    
    # 2. Preprocess input features
    input_data = request.features
    processed_features = {}
    
    # We will build a list of values ordered exactly by the hybrid_features list
    ordered_values = []
    
    # We will compute feature deviations for XAI
    contributions = []
    
    for col in hybrid_features:
        val = input_data.get(col)
        
        # Handle Missing values
        if val is None or val == "" or str(val).lower() == "nan":
            if col in categorical_mappings:
                # Fill missing categorical as "Missing"
                val = "Missing"
            else:
                # Fill missing numeric with median
                val = medians.get(col, 0.0)
        
        # Handle Categorical Encoding
        if col in categorical_mappings:
            categories = categorical_mappings[col]
            val_str = str(val).strip()
            if val_str in categories:
                encoded_val = categories.index(val_str)
            else:
                # Fallback to "Missing" category index if it exists, else index 0
                if "Missing" in categories:
                    encoded_val = categories.index("Missing")
                else:
                    encoded_val = 0
            processed_val = encoded_val
            
            # Simple categorical contribution heuristic
            # If the category is not the most common/default category, it deviates
            deviated = (val_str != categories[0])
            contrib = 1.0 if deviated else 0.0
        else:
            # Numeric Feature
            try:
                processed_val = float(val)
            except ValueError:
                processed_val = medians.get(col, 0.0)
            
            # Simple numeric contribution heuristic: deviation from median
            median_val = medians.get(col, 0.0)
            # Distance from median
            diff = processed_val - median_val
            # Simple scaling (to avoid dividing by zero)
            scale = abs(median_val) if median_val != 0 else 1.0
            contrib = diff / scale
            
        ordered_values.append(processed_val)
        processed_features[col] = processed_val
        
        # Save raw values and basic contribution before weighting with model importance
        contributions.append({
            "feature": col,
            "raw_value": val,
            "processed_value": processed_val,
            "raw_contrib": contrib
        })
        
    # Create 2D array for inference
    X_array = np.array([ordered_values])
    
    # 3. Model Inference
    probability = 0.0
    if engine == "xgboost":
        # Predict probability
        probs = xgb_model.predict_proba(X_array)
        probability = float(probs[0, 1])
        # Get XGBoost feature importances to weight XAI
        importances = xgb_model.feature_importances_
    else:  # lightgbm
        # LGBM booster predict directly returns probabilities
        probs = lgb_model.predict(X_array)
        probability = float(probs[0])
        # Get LightGBM feature importances
        # Standardize importances so they sum to 1
        raw_imps = lgb_model.feature_importance(importance_type='gain')
        total_imp = sum(raw_imps) if sum(raw_imps) > 0 else 1.0
        importances = [float(imp)/total_imp for imp in raw_imps]
        
    # 4. Apply Scorecard Overrides if sandbox
    is_sandbox = str(input_data.get("account_id", "")).startswith("SBX-")
    if is_sandbox:
        def safe_float(v, default=0.0):
            if v is None or v == "" or str(v).lower() == "nan":
                return default
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        def safe_int(v, default=0):
            if v is None or v == "" or str(v).lower() == "nan":
                return default
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return default

        def safe_str(v, default=""):
            if v is None:
                return default
            return str(v).strip()

        # Core Sandbox Features
        occ = safe_str(input_data.get("F3891", "")).lower()
        balance = safe_float(input_data.get("F3811", 0.0))
        freq = safe_float(input_data.get("F321", 0.0))
        velocity = safe_float(input_data.get("F2582", 0.0))
        acc_type = safe_str(input_data.get("F3886", "")).lower()
        branch = safe_int(input_data.get("F3887", 0))
        age = safe_float(input_data.get("F3894", 34.0))
        fraud_link = safe_float(input_data.get("F670", 0.0))
        
        # Extended Sandbox Features
        limit_ratio = safe_float(input_data.get("F115", 0.5))
        tx_act_a = safe_float(input_data.get("F527", 1.0))
        tx_act_b = safe_float(input_data.get("F531", 1.2))
        hr_transfers = safe_float(input_data.get("F1692", 0.0))
        assoc_a = safe_float(input_data.get("F2082", 0.0))
        assoc_b = safe_float(input_data.get("F2122", 0.0))
        dev_vel = safe_float(input_data.get("F2678", 0.0))
        loc_speed = safe_float(input_data.get("F2737", 0.0))
        off_hours = safe_float(input_data.get("F2956", 0.0))
        alert_freq = safe_float(input_data.get("F3043", 0.0))
        balance_net = safe_float(input_data.get("F3836", 0.0))
        active_win = safe_str(input_data.get("F3889", "G365D"))

        # Determine base features to check for manual perturbations
        base_features = request.base_features if request.base_features is not None else input_data
        
        occ_base = safe_str(base_features.get("F3891")).lower()
        occ_perturbed = occ != occ_base

        balance_base = safe_float(base_features.get("F3811"), 0.0)
        balance_perturbed = abs(balance - balance_base) > 1e-2

        freq_base = safe_float(base_features.get("F321"), 0.0)
        freq_perturbed = abs(freq - freq_base) > 1e-2

        velocity_base = safe_float(base_features.get("F2582"), 0.0)
        velocity_perturbed = abs(velocity - velocity_base) > 1e-2

        acc_type_base = safe_str(base_features.get("F3886")).lower()
        acc_type_perturbed = acc_type != acc_type_base

        branch_base = safe_int(base_features.get("F3887"), 0)
        branch_perturbed = branch != branch_base

        age_base = safe_float(base_features.get("F3894"), 34.0)
        age_perturbed = abs(age - age_base) > 1e-2

        fraud_link_base = safe_float(base_features.get("F670"), 0.0)
        fraud_link_perturbed = abs(fraud_link - fraud_link_base) > 1e-2

        limit_ratio_base = safe_float(base_features.get("F115"), 0.5)
        limit_ratio_perturbed = abs(limit_ratio - limit_ratio_base) > 1e-2

        tx_act_a_base = safe_float(base_features.get("F527"), 1.0)
        tx_act_a_perturbed = abs(tx_act_a - tx_act_a_base) > 1e-2

        tx_act_b_base = safe_float(base_features.get("F531"), 1.2)
        tx_act_b_perturbed = abs(tx_act_b - tx_act_b_base) > 1e-2

        hr_transfers_base = safe_float(base_features.get("F1692"), 0.0)
        hr_transfers_perturbed = abs(hr_transfers - hr_transfers_base) > 1e-2

        assoc_a_base = safe_float(base_features.get("F2082"), 0.0)
        assoc_a_perturbed = abs(assoc_a - assoc_a_base) > 1e-2

        assoc_b_base = safe_float(base_features.get("F2122"), 0.0)
        assoc_b_perturbed = abs(assoc_b - assoc_b_base) > 1e-2

        dev_vel_base = safe_float(base_features.get("F2678"), 0.0)
        dev_vel_perturbed = abs(dev_vel - dev_vel_base) > 1e-2

        loc_speed_base = safe_float(base_features.get("F2737"), 0.0)
        loc_speed_perturbed = abs(loc_speed - loc_speed_base) > 1e-2

        off_hours_base = safe_float(base_features.get("F2956"), 0.0)
        off_hours_perturbed = abs(off_hours - off_hours_base) > 1e-2

        alert_freq_base = safe_float(base_features.get("F3043"), 0.0)
        alert_freq_perturbed = abs(alert_freq - alert_freq_base) > 1e-2

        balance_net_base = safe_float(base_features.get("F3836"), 0.0)
        balance_net_perturbed = abs(balance_net - balance_net_base) > 1e-2

        active_win_base = safe_str(base_features.get("F3889"))
        active_win_perturbed = active_win != active_win_base

        # Check if mule baseline (Using F3898 as proxy)
        is_mule_baseline = (float(processed_features.get("F3898", 3.0)) < 2.0)
        
        if is_mule_baseline:
            # Mitigation mode
            mitigations = {}
            if occ_perturbed and occ == "salaried":
                mitigations["F3891"] = -0.25
            if balance_perturbed and balance < 100000:
                mitigations["F3811"] = -0.30
            if balance_net_perturbed and balance_net < 100000:
                mitigations["F3836"] = -0.30
            if freq_perturbed and freq < 3.0:
                mitigations["F321"] = -0.20
            if velocity_perturbed and velocity < 1.0:
                mitigations["F2582"] = -0.20
            if fraud_link_perturbed and fraud_link == 0.0:
                mitigations["F670"] = -0.25
            if hr_transfers_perturbed and hr_transfers < 1.0:
                mitigations["F1692"] = -0.15
            if alert_freq_perturbed and alert_freq < 50.0:
                mitigations["F3043"] = -0.15
            if loc_speed_perturbed and loc_speed < 5.0:
                mitigations["F2737"] = -0.15
            if limit_ratio_perturbed and limit_ratio < 0.2:
                mitigations["F115"] = -0.10
                
            total_mitigate = sum(mitigations.values())
            probability = max(0.01, probability + total_mitigate)
            
            # Weight contributions using Feature Importances
            for i, item in enumerate(contributions):
                imp = float(importances[i])
                col = item["feature"]
                if col in mitigations and mitigations[col] != 0.0:
                    item["contribution"] = mitigations[col]
                    item["raw_contrib"] = mitigations[col] / (imp + 1e-5)
                else:
                    item["contribution"] = item["raw_contrib"] * imp
                item["importance"] = imp
        else:
            # Risk escalation mode
            adjustments = {}
            
            # Occupation + balance rule
            if occ_perturbed or balance_perturbed or balance_net_perturbed:
                check_bal = balance_net if balance_net_perturbed else balance
                if occ in ["student", "housewife", "retired"]:
                    if check_bal > 1000000:
                        adjustments["F3836" if balance_net_perturbed else "F3811"] = 0.50
                    elif check_bal > 250000:
                        adjustments["F3836" if balance_net_perturbed else "F3811"] = 0.30
                    elif check_bal > 50000:
                        adjustments["F3836" if balance_net_perturbed else "F3811"] = 0.15
                elif occ in ["agriculture", "others"] and check_bal > 500000:
                    adjustments["F3836" if balance_net_perturbed else "F3811"] = 0.30
                
            # Savings frequency rule
            if freq_perturbed or acc_type_perturbed:
                if acc_type == "savings":
                    if freq > 10.0:
                        adjustments["F321"] = 0.25
                    elif freq > 5.0:
                        adjustments["F321"] = 0.15
                    
            # Velocity score rule
            if velocity_perturbed:
                if velocity > 3.0:
                    adjustments["F2582"] = 0.20
                elif velocity > 1.5:
                    adjustments["F2582"] = 0.10
                
            # Branch rule
            if branch_perturbed:
                if branch in [94, 98, 150]:
                    adjustments["F3887"] = 0.05
                    
            # Fraud Link Flag
            if fraud_link_perturbed and fraud_link == 1.0:
                adjustments["F670"] = 0.35
                
            # High-Risk Transfers
            if hr_transfers_perturbed:
                if hr_transfers > 5.0:
                    adjustments["F1692"] = 0.25
                elif hr_transfers > 2.0:
                    adjustments["F1692"] = 0.15
                    
            # Network association scores
            if assoc_a_perturbed and assoc_a > 0.8:
                adjustments["F2082"] = 0.15
            if assoc_b_perturbed and assoc_b > 0.8:
                adjustments["F2122"] = 0.15
                
            # Device velocity index
            if dev_vel_perturbed and dev_vel > 100000:
                adjustments["F2678"] = 0.25
                
            # Location speed
            if loc_speed_perturbed and loc_speed > 50.0:
                adjustments["F2737"] = 0.15
                
            # Off-hours logins
            if off_hours_perturbed and off_hours > 500:
                adjustments["F2956"] = 0.15
                
            # Alert frequency
            if alert_freq_perturbed and alert_freq > 500:
                adjustments["F3043"] = 0.15
                
            # Customer Age
            if age_perturbed and age < 23:
                adjustments["F3894"] = 0.10
                
            total_adjust = sum(adjustments.values())
            probability = min(0.99, probability + total_adjust)
            
            # Update contributions
            for i, item in enumerate(contributions):
                imp = float(importances[i])
                col = item["feature"]
                if col in adjustments:
                    item["contribution"] = adjustments[col]
                    item["raw_contrib"] = adjustments[col] / (imp + 1e-5)
                else:
                    item["contribution"] = item["raw_contrib"] * imp
                item["importance"] = imp
    else:
        # Standard weighting for non-sandbox accounts
        for i, item in enumerate(contributions):
            imp = float(importances[i])
            item["contribution"] = item["raw_contrib"] * imp
            item["importance"] = imp
            
    # Apply Custom Threshold
    prediction = 1 if probability >= request.threshold else 0
        
    # Sort contributions by absolute value descending to find top drivers
    sorted_contributions = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
    
    # Format top 5 drivers
    top_drivers = []
    for item in sorted_contributions[:5]:
        top_drivers.append({
            "feature": item["feature"],
            "value": item["raw_value"],
            "importance": round(item["importance"], 4),
            "contribution": round(item["contribution"], 4),
            # Plain English description
            "impact": "High Risk Driver" if item["contribution"] > 0 else "Mitigating Factor"
        })
        
    return {
        "account_id": input_data.get("account_id", "UNKNOWN"),
        "engine": engine,
        "probability": round(probability, 4),
        "prediction": prediction,
        "flagged": bool(prediction),
        "threshold": request.threshold,
        "top_drivers": top_drivers
    }

if __name__ == "__main__":
    import uvicorn
    import os
    # Bind to PORT if provided (Render), else default to localhost:8000 for development
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    reload = False if os.environ.get("PORT") else True
    uvicorn.run("app:app", host=host, port=port, reload=reload)
