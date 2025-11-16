# (Full file — paste over your current app.py)
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import Team_report_generator
import random
import string
import json
import traceback
from google import genai
import re

app = Flask(__name__)
CORS(app)

# --- NOTE: replace the API key with your secure secret in production ---
client = genai.Client(api_key="AIzaSyBJ6_G2P023DaJfVStaV6PyA3OqKTrSA_0")


def load_all_predictions(home, away):
    """Load full precomputed predictions JSON."""
    fname = f"predictions_{home}_{away}.json"
    fpath = os.path.join("assets", fname)

    if not os.path.exists(fpath):
        print("[ERROR] Prediction JSON missing:", fpath)
        return None

    try:
        with open(fpath, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Failed to load predictions JSON:", e)
        return None


# Route for serving PDF and JSON reports
@app.route("/reports/<path:filename>")
def serve_reports(filename):
    reports_dir = os.path.join(os.getcwd(), "assets")
    return send_from_directory(reports_dir, filename)


# Sample placeholder text (legacy, used by reinforce endpoint)
LOREM_LIST = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse.",
    "Cillum dolore eu fugiat nulla pariatur.",
    "Excepteur sint occaecat cupidatat non proident.",
    "Sunt in culpa qui officia deserunt mollit anim id est laborum."
]


@app.get("/pregenerate")
def pregenerate():
    home = request.args.get("home")
    away = request.args.get("away")

    if not home or not away:
        return jsonify({"error": "Missing team parameters"}), 400

    home = home.capitalize()
    away = away.capitalize()
    print(f"🏁 Starting generation for {home} vs {away}")

    # === TEAM REPORT ===
    try:
        from Team_report_generator import reportGenerator
        status = reportGenerator(homeTeam=home, awayTeam=away)
        if status < 0:
            print("❌ Team report generation failed.")
            return jsonify({"error": "Error in generating team report"}), 500
        else:
            print("✅ Team report generated successfully.")
    except Exception:
        print("❌ Team report crashed:")
        traceback.print_exc()
        return jsonify({"error": "Team report failed"}), 500

    # === PLAYER REPORTS ===
    try:
        from Player_report_generator import load_data, PerformancePredictor, ImpactAnalyzer, create_report

        data_file = os.path.join("assets", "Premier_League.csv")
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 Dataset path: {os.path.abspath(data_file)}")

        data = load_data(data_file)
        if data is None:
            print("❌ Dataset not loaded — skipping player reports.")
        else:
            predictor = PerformancePredictor()
            analyzer = ImpactAnalyzer()

            print(f"🧩 Creating report for {home}")
            create_report(home, away, data, predictor, analyzer, output_dir=output_dir)
            print(f"🧩 Creating report for {away}")
            create_report(away, home, data, predictor, analyzer, output_dir=output_dir)

            print("✅ Player reports created successfully.")
    except Exception:
        print("❌ Player report generation crashed:")
        traceback.print_exc()

    # === PREDICTIONS JSON ===
    try:
        from prediction_utils import precompute_predictions
        from data_provider import DataProvider
        from discipline import DisciplineModel
        import math

        print("🧠 Generating predictions JSON...")
        print("Current working directory:", os.getcwd())
        print("Assets path:", os.path.abspath("assets"))

        discipline_model = DisciplineModel(file_path=os.path.join("assets", "E0.csv"))

        class DummyModel:
            def __init__(self, name): self.name = name
            def predict_proba(self, X):
                t = X[0][0] if X and isinstance(X[0], (list, tuple)) else random.random()
                p = 0.4 + 0.3 * math.sin(t / 300) + random.uniform(-0.05, 0.05)
                p = max(0, min(1, p))
                return [[1 - p, p]]

        attack_model = DummyModel("attack_defend")
        formation_model = DummyModel("formation")
        weakness_model = DummyModel("weakness")
        pressing_model = DummyModel("pressing")
        substitution_model = DummyModel("substitution")

        data_file = os.path.join("assets", "E0.csv")
        data_provider = DataProvider(data_file)

        # FIXED small typo here: keep correct variable names
        predictors = {
            "attack_defend": attack_model,
            "discipline": discipline_model,
            "formation": formation_model,
            "weakness": weakness_model,
            "pressing": pressing_model,
            "substitution": substitution_model
        }

        pred_path = precompute_predictions(
            home, away, data_provider, predictors,
            output_dir="assets", step=1
        )

        if not pred_path or not os.path.exists(pred_path):
            print("⚠️ precompute_predictions() did not create a file, generating fallback JSON...")
            fallback_path = os.path.join("assets", f"predictions_{home}_{away}.json")
            dummy_json = {
                "match_meta": {"home": home, "away": away, "duration_sec": 5400},
                "models": {
                    "discipline": {"label": "Discipline", "values": [0.1, 0.2, 0.3]},
                    "attack_defend": {"label": "Attack/Defend", "values": [0.4, 0.5, 0.6]},
                    "formation": {"label": "Formation", "values": [0.5, 0.6, 0.7]},
                    "weakness": {"label": "Weakness", "values": [0.2, 0.3, 0.4]},
                    "pressing": {"label": "Pressing", "values": [0.3, 0.4, 0.5]},
                    "substitution": {"label": "Substitution", "values": [0.1, 0.2, 0.3]}
                }
            }
            with open(fallback_path, "w") as f:
                json.dump(dummy_json, f, indent=2)
            pred_path = fallback_path
            print(f"✅ Fallback predictions saved at {pred_path}")
        else:
            print(f"✅ Predictions saved at {pred_path}")

    except Exception:
        print("❌ Prediction JSON generation failed:")
        traceback.print_exc()

    team_path = f"assets/{home.lower()}-{away.lower()}.pdf"
    print(f"📄 Returning {team_path}")
    if not os.path.exists(team_path):
        print(f"❌ Team PDF not found at {team_path}")
        return jsonify({"error": "Team report missing"}), 500

    resp = send_file(team_path, mimetype="application/pdf", as_attachment=False)
    try:
        resp.set_cookie("home_team", home)
        resp.set_cookie("away_team", away)
        return resp
    except Exception:
        print("❌ Cannot set cookies")


# =========================================================
# =============== LIVE PREDICTIONS (PATCHED & ROBUST) ======
# =========================================================
@app.route("/live_predictions", methods=["POST"])
def livepred():
    """
    Expects POST JSON: { "min": <int minutes> }
    This route uses BUILT-IN Chelsea vs Arsenal 2022-23 timeline only (per your request A).
    Returns: { "models": [6 paragraphs], "summary": paragraph7 }
    """

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid json"}), 400

    minute = int(payload.get("min", 0))

    # -------------------------------------------------------
    # BUILT-IN MATCH EVENTS (Chelsea vs Arsenal 2022-23)
    # -------------------------------------------------------
    builtin_events = [
        {"minute": 0, "event": "Kick-off"},
        {"minute": 14, "event": "Chance: Arsenal – Gabriel Jesus header misses the target"},
        {"minute": 28, "event": "Foul: Mason Mount (Chelsea) brings down Thomas Partey"},
        {"minute": 32, "event": "Shot: Chelsea – Raheem Sterling shot blocked inside the box"},
        {"minute": 42, "event": "Yellow card: Bukayo Saka (Arsenal) for a tactical foul"},
        {"minute": 45, "event": "Half-time: Chelsea 0 – 0 Arsenal"},
        {"minute": 51, "event": "Yellow card: César Azpilicueta (Chelsea) for a late challenge"},
        {"minute": 55, "event": "High pressing: Arsenal increasing pressure"},
        {"minute": 62, "event": "Chance: Arsenal – Gabriel Jesus shot saved by Mendy"},
        {"minute": 63, "event": "GOAL: Gabriel Magalhães (Arsenal) scores from a corner"},
        {"minute": 64, "event": "Substitution: Gallagher and Broja on for Chelsea"},
        {"minute": 72, "event": "Chance: Arsenal – Martinelli shoots wide"},
        {"minute": 78, "event": "Substitution: Tierney replaces Zinchenko"},
        {"minute": 85, "event": "Yellow card: Conor Gallagher (Chelsea) for dissent"},
        {"minute": 89, "event": "Foul: Raheem Sterling (Chelsea) commits dangerous challenge"},
        {"minute": 90, "event": "Full time: Chelsea 0 – 1 Arsenal"}
    ]

    past_events = [ev for ev in builtin_events if ev["minute"] <= minute]

    # -------------------------------------------------------
    # Load prediction JSON (expected format from pregenerate)
    # -------------------------------------------------------
    preds_path = os.path.join("assets", "predictions_chelsea_arsenal.json")
    if not os.path.exists(preds_path):
        # fallback: try other common filename
        preds_path = os.path.join("assets", f"predictions_Chelsea_Arsenal.json")
    if not os.path.exists(preds_path):
        # if still missing, create safe defaults
        models_data = {
            "attack_defend": {"label": "Attack/Defend", "values": [0.45]*100},
            "discipline": {"label": "Discipline", "values": [0.2]*100},
            "formation": {"label": "Formation", "values": [0.5]*100},
            "weakness": {"label": "Weakness", "values": [0.3]*100},
            "pressing": {"label": "Pressing", "values": [0.3]*100},
            "substitution": {"label": "Substitution", "values": [0.1]*100}
        }
    else:
        try:
            with open(preds_path, "r") as f:
                j = json.load(f)
                models_data = j.get("models", {})
        except Exception as e:
            print("Error loading predictions JSON:", e)
            models_data = {
                "attack_defend": {"label": "Attack/Defend", "values": [0.45]*100},
                "discipline": {"label": "Discipline", "values": [0.2]*100},
                "formation": {"label": "Formation", "values": [0.5]*100},
                "weakness": {"label": "Weakness", "values": [0.3]*100},
                "pressing": {"label": "Pressing", "values": [0.3]*100},
                "substitution": {"label": "Substitution", "values": [0.1]*100}
            }

    # Build numeric vals at this minute
    numeric_vals = {}
    for key in ["attack_defend", "discipline", "formation", "weakness", "pressing", "substitution"]:
        model = models_data.get(key, {})
        vals = model.get("values", [])
        if not vals:
            numeric_vals[key] = 0.0
        else:
            idx = min(len(vals) - 1, int(minute))
            try:
                numeric_vals[key] = float(vals[idx])
            except:
                numeric_vals[key] = float(vals[-1])

    # -------------------------------------------------------
    # EVENT-DRIVEN SPIKES (Makes model "predict" events)
    # -------------------------------------------------------
    for ev in past_events:
        text = ev["event"].lower()

        if "goal" in text:
            numeric_vals["attack_defend"] = max(numeric_vals.get("attack_defend", 0.0), 0.9)
            numeric_vals["pressing"] = max(numeric_vals.get("pressing", 0.0), 0.8)

        if "yellow" in text:
            numeric_vals["discipline"] = max(numeric_vals.get("discipline", 0.0), 0.85)

        if "foul" in text:
            numeric_vals["discipline"] = max(numeric_vals.get("discipline", 0.0), 0.75)

        if "shot" in text or "chance" in text:
            numeric_vals["attack_defend"] = max(numeric_vals.get("attack_defend", 0.0), 0.75)

        if "press" in text or "pressing" in text:
            numeric_vals["pressing"] = max(numeric_vals.get("pressing", 0.0), 0.8)

        if "substitution" in text:
            numeric_vals["substitution"] = max(numeric_vals.get("substitution", 0.0), 0.9)

        if "counter" in text:
            numeric_vals["weakness"] = max(numeric_vals.get("weakness", 0.0), 0.8)

    # -------------------------------------------------------
    # INJECT SPIKED VALUES INTO models_data (so frontend sees spikes)
    # -------------------------------------------------------
    for k, val in numeric_vals.items():
        if k in models_data:
            vals = models_data[k].get("values", [])
            if vals:
                idx = min(len(vals) - 1, int(minute))
                try:
                    vals[idx] = float(val)
                except:
                    vals[idx] = val
            else:
                # ensure the values array exists and is long enough
                models_data[k]["values"] = [float(val)] * (int(minute) + 1)

    # -------------------------------------------------------
    # Prepare the raw summary for Gemini
    # -------------------------------------------------------
    model_labels = {
        "attack_defend": models_data.get("attack_defend", {}).get("label", "Attack/Defend"),
        "discipline": models_data.get("discipline", {}).get("label", "Discipline"),
        "formation": models_data.get("formation", {}).get("label", "Formation"),
        "weakness": models_data.get("weakness", {}).get("label", "Weakness"),
        "pressing": models_data.get("pressing", {}).get("label", "Pressing"),
        "substitution": models_data.get("substitution", {}).get("label", "Substitution")
    }

    raw_lines = [f"minute: {minute}"]
    for k in ["attack_defend", "discipline", "formation", "weakness", "pressing", "substitution"]:
        raw_lines.append(f"{model_labels.get(k, k)}: {numeric_vals.get(k, 0.0):.2f}")
    raw_summary = "\n".join(raw_lines)

    # Build a string of past events for the prompt
    try:
        match_events_text = json.dumps(past_events, ensure_ascii=False, indent=2)
    except:
        match_events_text = str(past_events)

    # -------------------------------------------------------
    # Prompt to Gemini
    # -------------------------------------------------------
    prompt = (
        "You are a professional football analyst. Produce exactly 7 short paragraphs separated by the delimiter _^^_.\n"
        "Each paragraph must start with a single <h5>heading</h5> tag for the model name (or 'Combined Summary') and then 1-3 short sentences.\n"
        "Order: attack_defend, discipline, formation, weakness, pressing, substitution, Combined Summary.\n\n"
        f"Past match events up to this minute:\n{match_events_text}\n\n"
        f"Current minute and model snapshot:\n{raw_summary}\n\n"
        "Rules:\n"
        "- If a model value >= 0.70, prepend ALERTALERT and a single space before the <h5> tag for that paragraph.\n"
        "- Keep paragraphs short. Use concrete tactical language. If a 'goal' is present in events, mention 'goal' and escalate attack/pressing language.\n"
        "- Output exactly 7 paragraphs separated by _^^_ (no leading/trailing separators). Each paragraph must include a <h5> heading.\n"
    )

    # -------------------------------------------------------
    # Call Gemini
    # -------------------------------------------------------
    summary_text = ""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        summary_text = response.text or ""
    except Exception as e:
        print("Gemini call failed:", e)
        summary_text = ""

    # -------------------------------------------------------
    # Parse and sanitize Gemini response into 7 paragraphs
    # -------------------------------------------------------
    # Split on the exact delimiter
    parts = [p.strip() for p in summary_text.split("_^^_") if p.strip() != ""]

    # Collapse repeating ALERTALERTs and normalize whitespace & ensure single ALERTALERT prefix
    def sanitize_paragraph(p):
        # collapse repeated ALERTALERT occurrences
        p = re.sub(r'(ALERTALERT\s*)+', 'ALERTALERT ', p, flags=re.IGNORECASE).strip()
        # ensure only one leading ALERTALERT and normalize
        if p.upper().startswith("ALERTALERT"):
            # ensure uppercase and single spacing
            p = "ALERTALERT " + p[len("ALERTALERT"):].lstrip()
        return p

    parts = [sanitize_paragraph(p) for p in parts]

    # If we got exactly 7 parts and each contains a <h5> we accept them.
    valid = len(parts) == 7 and all("<h5>" in p and "</h5>" in p for p in parts)

    paragraphs = []
    if valid:
        paragraphs = parts
    else:
        # Attempt to salvage: if we have 7 parts but missing headings, inject headings in order
        if len(parts) == 7:
            ordered_keys = ["attack_defend", "discipline", "formation", "weakness", "pressing", "substitution", "combined"]
            newp = []
            for i, p in enumerate(parts):
                if "<h5>" not in p:
                    heading = model_labels.get(ordered_keys[i], ordered_keys[i].title()) if i < 6 else "Combined Summary"
                    p = f"<h5>{heading}</h5>{p}"
                newp.append(sanitize_paragraph(p))
            paragraphs = newp
        else:
            # Not enough usable parts — build deterministic paragraphs from numeric_vals
            def make_par(title, txt, alert=False):
                pref = "ALERTALERT " if alert else ""
                return f"{pref}<h5>{title}</h5>{txt}"

            ad = numeric_vals.get("attack_defend", 0.0)
            di = numeric_vals.get("discipline", 0.0)
            fo = numeric_vals.get("formation", 0.0)
            we = numeric_vals.get("weakness", 0.0)
            pr = numeric_vals.get("pressing", 0.0)
            su = numeric_vals.get("substitution", 0.0)

            paragraphs = [
                make_par("Attack/Defend", f"Attacking tendency {(ad*100):.0f}%. {'Strong chance present.' if ad>=0.8 else 'No immediate clear chance.'}", alert=(ad>=0.7)),
                make_par("Discipline", f"Discipline {(di*100):.0f}%. {'Elevated risk of cards.' if di>=0.6 else 'Discipline looks okay.'}", alert=(di>=0.7)),
                make_par("Formation", f"Tactical stability {(fo*100):.0f}%. Team shape {'stable' if fo>=0.6 else 'fluid'}.", alert=(fo<=0.25)),
                make_par("Weakness", f"Vulnerability {(we*100):.0f}% to opponent exploitation.", alert=(we>=0.6)),
                make_par("Pressing", f"Pressing intensity {(pr*100):.0f}%. {'High pressure' if pr>=0.65 else 'Pressure moderate.'}", alert=(pr>=0.7)),
                make_par("Substitution", f"Substitution likelihood {(su*100):.0f}%. {'Coach likely to change.' if su>=0.5 else 'Stable.'}", alert=(su>=0.6)),
                make_par("Combined Summary", f"Overall: attack {ad:.2f}, press {pr:.2f}, discipline {di:.2f}.")
            ]

    # Ensure we have exactly 7 items
    if len(paragraphs) > 7:
        paragraphs = paragraphs[:7]
    while len(paragraphs) < 7:
        paragraphs.append(f"<h5>Combined Summary</h5>")

    # Final sanitize: ensure each paragraph is a trimmed string
    paragraphs = [str(p).strip() for p in paragraphs]

    # Return 6 model paragraphs + final summary
    return jsonify({
        "models": paragraphs[:6],
        "summary": paragraphs[6]
    })


@app.post("/reinforce")
def reinforce():
    data = request.json or {}
    index = data.get("index")
    events = data.get("matchEvents")
    timeStamp = data.get("timeStamp")

    if not isinstance(index, int) or not isinstance(events, list):
        return jsonify({"error": "index (int) and matchEvents (array) required"}), 400

    for i in range(0, len(LOREM_LIST)):
        if i == index:
            LOREM_LIST[i] += f"\nreinforced self at " + timeStamp
        else:
            j = i + 1
            LOREM_LIST[i] += f"\nreinforced model " + str(j) + " at " + timeStamp
    return jsonify(LOREM_LIST)


if __name__ == "__main__":
    app.run(debug=True)
