import pandas as pd
from joblib import dump
from attack_model import AttackModel
from discipline_model import DisciplineModel
from formation_model import FormationModel
from weakness_model import WeaknessModel
from pressing_model import PressingModel
from substitution_model import SubstitutionModel

# 1️⃣ Load your dataset
df = pd.read_csv("assets/Premier_League.csv")

# 2️⃣ Define feature columns and target columns
# Example: adjust these based on your actual data
feature_cols = ["possession", "shots", "fouls", "corners", "xg"]
target_col_attack = "attack_label"       # Adjust based on your CSV
target_col_discipline = "discipline_label"
target_col_formation = "formation_label"
target_col_weakness = "weakness_label"
target_col_pressing = "pressing_label"
target_col_substitution = "substitution_label"

X = df[feature_cols]

# 3️⃣ Create model folder if not exists
import os
os.makedirs("models", exist_ok=True)

# 4️⃣ Train and save each model
def train_and_save(model_class, y_col, filename):
    model = model_class()
    y = df[y_col]
    model.train(X, y)
    dump(model.model, f"models/{filename}.pkl")
    print(f"✅ Saved trained {filename}.pkl")

train_and_save(AttackModel, target_col_attack, "attack_model")
train_and_save(DisciplineModel, target_col_discipline, "discipline_model")
train_and_save(FormationModel, target_col_formation, "formation_model")
train_and_save(WeaknessModel, target_col_weakness, "weakness_model")
train_and_save(PressingModel, target_col_pressing, "pressing_model")
train_and_save(SubstitutionModel, target_col_substitution, "substitution_model")

print("🎉 All models trained and saved successfully.")
