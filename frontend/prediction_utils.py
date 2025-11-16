# prediction_utils.py
import os, json
import numpy as np

# Put model label and threshold mappings here (tweak thresholds per model)
MODEL_LABELS = {
    "attack_defend": "Attack/Defend",
    "discipline": "Discipline",
    "formation": "Formation",
    "weakness": "Weakness",
    "pressing": "Pressing",
    "substitution": "Substitution"
}
MODEL_THRESHOLDS = {
    "attack_defend": 0.7,
    "discipline": 0.6,
    "formation": 0.65,
    "weakness": 0.6,
    "pressing": 0.7,
    "substitution": 0.55
}

def precompute_predictions(home, away, data_provider, predictors, output_dir="assets", step=1):
    """
    home, away : capitalized team names
    data_provider: an object or module with function features_at_time(t) and match_events()
    predictors: dict mapping model_key -> trained_model_object
    output_dir: where to save JSON
    step: seconds per data point (1 recommended)
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = 90*60  # 90 minutes in seconds
    times = list(range(0, duration+1, step))

    out = {
        "match_meta": {"home": home, "away": away, "duration_sec": duration},
        "time_step": step,
        "models": {},
        "events": []
    }

    # For each model generate probability/time series
    for key, model in predictors.items():
        vals = []
        for t in times:
            X = data_provider.features_at_time(t, home, away)  # implement this to return features
            # prefer predict_proba if available
            try:
                prob = model.predict_proba([X])[0]
                # if binary, take positive class
                if len(prob) > 1:
                    vals.append(float(prob[1]))
                else:
                    vals.append(float(prob[0]))
            except Exception:
                # fallback to decision_function or predict
                try:
                    vals.append(float(model.decision_function([X])[0]))
                except Exception:
                    vals.append(float(model.predict([X])[0]))

        out["models"][key] = {
            "label": MODEL_LABELS.get(key, key),
            "threshold": MODEL_THRESHOLDS.get(key, 0.7),
            "values": vals
        }

    # events (if your data_provider supplies them)
    try:
        out["events"] = data_provider.match_events(home, away)  # list of {"time":..., "type":..., "desc":...}
    except Exception:
        out["events"] = []

    filepath = os.path.join(output_dir, f"predictions_{home.lower()}_{away.lower()}.json")
    with open(filepath, "w") as f:
        json.dump(out, f)
    print("Saved predictions to", filepath)
    return filepath
