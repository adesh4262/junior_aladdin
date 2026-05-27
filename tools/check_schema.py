import json, sys
sys.path.insert(0, ".")
from collections import deque
from src.core.replay_engine import ReplayEngine
from src.features.feature_engine import FeatureEngine

DATE = "2026-03-25"

# Load historical candles
replay = ReplayEngine()
replay.load(DATE)
candles_list = list(replay._candles)

# Build minimal candles_by_tf dict expected by FeatureEngine
candles_by_tf = {
    "1min": deque(candles_list, maxlen=400),
    "3min": deque(maxlen=140),
    "5min": deque(maxlen=80),
    "15min": deque(maxlen=30),
}

engine = FeatureEngine()
bundle = engine.compute_all(
    candles_by_tf=candles_by_tf,
    option_chain={},
    market_depth={},
    spot_price=24500,
    previous_day_candles=None,
)

# --- Extract actual feature keys ---
features_1m = bundle.get("per_tf", {}).get("1min", {})
print("--- ACTUAL keys in features_1m (from FeatureEngine) ---")
for k in sorted(features_1m.keys()):
    print(f"  {k}")

print("\n--- ACTUAL keys in volume_profile ---")
vp = bundle.get("volume_profile", {})
for k in sorted(vp.keys()):
    print(f"  {k}")

print("\n--- ACTUAL keys in key_levels ---")
kl = bundle.get("key_levels", {})
for k in sorted(kl.keys()):
    print(f"  {k}")

# --- Compare with schema ---
with open("models/lightgbm_quality_filter_feature_schema.json") as f:
    schema = json.load(f)["features"]

print("\n=== SCHEMA MISMATCHES (features_1m) ===")
for item in schema:
    if item["source"] == "features_1m":
        p = item["path"]
        if p in features_1m:
            print(f'  MATCH: "{p}"')
        else:
            similar = [k for k in features_1m if p in k or k in p]
            print(f'  MISMATCH: expects "{p}" -> NOT FOUND in features_1m. Similar keys: {similar}')