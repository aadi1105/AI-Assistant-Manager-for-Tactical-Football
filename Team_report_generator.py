#!/usr/bin/env python
#
# AI-Powered Football Scout Report Generator
#
# This single-file script loads and processes football match data (E0.csv),
# re-implements and trains five separate AI models, and generates a multi-page
# PDF scout report comparing two user-selected teams.
#

# ----------------------------------------------------------------------
# 1. SETUP & IMPORTS
# ----------------------------------------------------------------------
import sys
import os
import warnings
import datetime
import math
import joblib
from typing import Dict, List, Tuple, Any, Optional

# Data & Processing
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import NotFittedError

# Plotting
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

# PDF Generation
from fpdf import FPDF
from fpdf.enums import XPos, YPos # Import new enums for PDF generation

# User Input
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# Global Configuration
RANDOM_STATE = 42
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)
# *** FIX HERE: Suppress the DeprecationWarnings from fpdf2 ***
warnings.filterwarnings('ignore', category=DeprecationWarning, module='fpdf') 
CHART_DIR = "charts"


# ----------------------------------------------------------------------
# 2. GLOBAL HELPER FUNCTIONS
# ----------------------------------------------------------------------

def load_df(path: str) -> pd.DataFrame:
    """
    Loads a CSV file with multiple encoding attempts.
    """
    # Use the path provided, or default if it's just the filename
    csv_file_path = path
    if not os.path.isfile(csv_file_path):
        # Try to find it in a common relative path as seen in the user's error
        alt_path = os.path.join(os.path.dirname(__file__), "..", "Feature selection", "E0.csv")
        if os.path.isfile(alt_path):
            csv_file_path = alt_path
        elif os.path.isfile("E0.csv"):
             csv_file_path = "E0.csv"
        else:
            print(f"Error: Could not find '{path}' or in default locations.")
            
    
    for e in ["utf-8", "ISO-8859-1", "latin1"]:
        try:
            return pd.read_csv(csv_file_path, encoding=e)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read {csv_file_path} with any common encoding.")

def safe_series(s: pd.Series) -> pd.Series:
    """
    Converts a Series to numeric, filling NaNs with 0.
    """
    return pd.to_numeric(s, errors="coerce").fillna(0)

def require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    """
    Checks if all required columns exist in the DataFrame.
    """
    miss = [c for c in cols if c not in df.columns]
    if miss:
        # Check for common alternatives (e.g., 'HBP' vs 'B365HBP')
        # This is a basic check; a real-world scenario might need more mapping
        if 'HBP' in miss and 'B365HBP' in df.columns:
            df.rename(columns={'B365HBP': 'HBP'}, inplace=True)
            miss.remove('HBP')
        if 'ABP' in miss and 'B365ABP' in df.columns:
            df.rename(columns={'B365ABP': 'ABP'}, inplace=True)
            miss.remove('ABP')
        
        if miss:
            print(f"Warning: Missing columns, AI accuracy may be affected: {miss}", file=sys.stderr)
            # Add missing columns with 0s to prevent crashes
            for c in miss:
                df[c] = 0

def compute_booking_points_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Home Booking Points (HBP) and Away Booking Points (ABP)
    if they are not present in the dataset.
    Yellow card = 10 points, Red card = 25 points.
    """
    if 'HBP' not in df.columns and 'HY' in df.columns and 'HR' in df.columns:
        print("Calculating 'HBP' from 'HY' and 'HR'.")
        df['HBP'] = safe_series(df['HY']) * 10 + safe_series(df['HR']) * 25
    
    if 'ABP' not in df.columns and 'AY' in df.columns and 'AR' in df.columns:
        print("Calculating 'ABP' from 'AY' and 'AR'.")
        df['ABP'] = safe_series(df['AY']) * 10 + safe_series(df['AR']) * 25
    
    # Ensure columns exist even if card data was missing
    if 'HBP' not in df.columns: df['HBP'] = 0
    if 'ABP' not in df.columns: df['ABP'] = 0
        
    return df


# ----------------------------------------------------------------------
# 3. THE "AI CORE" (MODEL CLASSES)
# ----------------------------------------------------------------------

class AttackDefendAI:
    """
    AI Model 1: Analyzes "Push for Goal" vs. "Defend Lead".
    Predicts if the match outcome will be a Home win (Defend) or Away win (Push).
    """
    def __init__(self, random_state=RANDOM_STATE):
        self.model = RandomForestClassifier(random_state=random_state, n_estimators=50)
        self.features = ["HS", "AS", "HST", "AST", "HC", "AC", "FTHG", "FTAG", "HTHG", "HTAG"]
        self.is_trained = False

    def _build_features(self, df: pd.DataFrame, safe_s: callable) -> pd.DataFrame:
        require_cols(df, self.features)
        X = pd.DataFrame(index=df.index)
        for c in self.features:
            X[c] = safe_s(df[c])
        return X

    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        require_cols(df, ["FTR"])
        # 0 = Home Win ("Defend Lead"), 1 = Away Win ("Push for Goal")
        y = df['FTR'].map({'H': 0, 'A': 1})
        return y

    def train(self, df: pd.DataFrame, safe_s: callable):
        y = self._build_target(df)
        X = self._build_features(df[y.notna()], safe_s)
        y = y[y.notna()]
        
        if len(X) < 10:
            print("Warning (AttackDefendAI): Not enough data to train.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"AttackDefendAI trained. Accuracy: {self.model.score(X_test, y_test):.2f}")

    def get_team_averages(self, df: pd.DataFrame, safe_s: callable) -> Tuple[pd.DataFrame, pd.DataFrame]:
        home_cols = ["HS", "HST", "HC", "HF", "HY", "HR", "FTHG", "HTHG"]
        away_cols = ["AS", "AST", "AC", "AF", "AY", "AR", "FTAG", "HTAG"]
        require_cols(df, home_cols + away_cols)
        
        home_avg = df.groupby("HomeTeam")[home_cols].mean()
        away_avg = df.groupby("AwayTeam")[away_cols].mean()
        
        return home_avg, away_avg

    def build_match_row(self, team_avgs: Tuple[pd.DataFrame, pd.DataFrame], home_team: str, away_team: str) -> Optional[pd.DataFrame]:
        home_avg, away_avg = team_avgs
        try:
            home_row = home_avg.loc[home_team]
            away_row = away_avg.loc[away_team]
        except KeyError:
            print(f"Warning (AttackDefendAI): Missing avg data for {home_team} or {away_team}")
            return None
        
        Xone = pd.DataFrame(index=[0])
        Xone["HS"] = home_row["HS"]
        Xone["AS"] = away_row["AS"]
        Xone["HST"] = home_row["HST"]
        Xone["AST"] = away_row["AST"]
        Xone["HC"] = home_row["HC"]
        Xone["AC"] = away_row["AC"]
        Xone["FTHG"] = home_row["FTHG"]
        Xone["FTAG"] = away_row["FTAG"]
        Xone["HTHG"] = home_row["HTHG"]
        Xone["HTAG"] = away_row["HTAG"]
        return Xone[self.features] # Ensure column order

    def get_prediction_and_insights(self, Xone: pd.DataFrame) -> Tuple[str, List[str]]:
        if not self.is_trained:
            return "Model not trained.", ["Insufficient data."]
            
        pred = self.model.predict(Xone)[0]
        prob = self.model.predict_proba(Xone)[0]
        
        prediction_text = (
            f"AI Recommends: PUSH FOR GOAL (Away Win Prob: {prob[1]:.0%})" if pred == 1 
            else f"AI Recommends: DEFEND/HOLD (Home Win Prob: {prob[0]:.0%})"
        )
        
        insights = []
        if Xone['HS'].iloc[0] > Xone['AS'].iloc[0] * 1.2:
            insights.append(f"Home's higher shot volume ({Xone['HS'].iloc[0]:.1f}) supports an aggressive stance.")
        if Xone['AST'].iloc[0] > Xone['HST'].iloc[0] * 1.2:
            insights.append(f"Away's superior accuracy ({Xone['AST'].iloc[0]:.1f} SOT) is a key threat.")
        if Xone['FTHG'].iloc[0] < Xone['FTAG'].iloc[0]:
            insights.append(f"Away's better goal record ({Xone['FTAG'].iloc[0]:.1f}/game) suggests they are favorites.")
        else:
            insights.append(f"Home's strong goal record ({Xone['FTHG'].iloc[0]:.1f}/game) provides a solid foundation.")
            
        summary = "Focus on exploiting the imbalance in shots on target."
        if pred == 1:
            summary = "Away team should leverage their clinical finishing. Home must increase defensive pressure."
        else:
            summary = "Home team should control the pace, leveraging their higher shot volume."

        return prediction_text, insights, summary


class DisciplineAI:
    """
    AI Model 2: Analyzes "Manage Player Discipline".
    Predicts which team is likely to be more disciplined (fewer booking points).
    """
    WEIGHTS = {"foul_point": 1.0, "yellow_point": 5.0, "red_point": 15.0}
    
    def __init__(self, random_state=RANDOM_STATE):
        self.model = RandomForestClassifier(random_state=random_state, n_estimators=50)
        self.features = ["Foul_Diff", "Yellow_Diff", "Red_Diff"]
        self.is_trained = False

    def _build_features(self, df: pd.DataFrame, safe_s: callable) -> pd.DataFrame:
        require_cols(df, ["HF", "AF", "HY", "AY", "HR", "AR"])
        X = pd.DataFrame(index=df.index)
        X["Foul_Diff"] = safe_s(df["HF"]) - safe_s(df["AF"])
        X["Yellow_Diff"] = safe_s(df["HY"]) - safe_s(df["AY"])
        X["Red_Diff"] = safe_s(df["HR"]) - safe_s(df["AR"])
        return X

    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        require_cols(df, ["HBP", "ABP"])
        # 1 = Home more disciplined (HBP < ABP), 0 = Away more disciplined (HBP >= ABP)
        y = (safe_series(df["HBP"]) < safe_series(df["ABP"])).astype(int)
        return y

    def train(self, df: pd.DataFrame, safe_s: callable):
        y = self._build_target(df)
        X = self._build_features(df, safe_s)
        
        if len(X) < 10:
            print("Warning (DisciplineAI): Not enough data to train.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"DisciplineAI trained. Accuracy: {self.model.score(X_test, y_test):.2f}")

    def get_team_averages(self, df: pd.DataFrame, safe_s: callable) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cols = ["HF", "HY", "HR", "AF", "AY", "AR"]
        require_cols(df, cols)
        
        home_avg = df.groupby("HomeTeam").agg(
            Fouls_Avg=("HF", "mean"), 
            Yellows_Avg=("HY", "mean"), 
            Reds_Avg=("HR", "mean")
        )
        away_avg = df.groupby("AwayTeam").agg(
            Fouls_Avg=("AF", "mean"), 
            Yellows_Avg=("AY", "mean"), 
            Reds_Avg=("AR", "mean")
        )
        
        home_avg["disc_score"] = (
            home_avg["Fouls_Avg"] * self.WEIGHTS["foul_point"] +
            home_avg["Yellows_Avg"] * self.WEIGHTS["yellow_point"] +
            home_avg["Reds_Avg"] * self.WEIGHTS["red_point"]
        )
        away_avg["disc_score"] = (
            away_avg["Fouls_Avg"] * self.WEIGHTS["foul_point"] +
            away_avg["Yellows_Avg"] * self.WEIGHTS["yellow_point"] +
            away_avg["Reds_Avg"] * self.WEIGHTS["red_point"]
        )
        return home_avg, away_avg

    def build_match_row(self, team_avgs: Tuple[pd.DataFrame, pd.DataFrame], home_team: str, away_team: str) -> Optional[pd.DataFrame]:
        home_avg, away_avg = team_avgs
        try:
            home_row = home_avg.loc[home_team]
            away_row = away_avg.loc[away_team]
        except KeyError:
            print(f"Warning (DisciplineAI): Missing avg data for {home_team} or {away_team}")
            return None
            
        Xone = pd.DataFrame(index=[0])
        Xone["Foul_Diff"] = home_row["Fouls_Avg"] - away_row["Fouls_Avg"]
        Xone["Yellow_Diff"] = home_row["Yellows_Avg"] - away_row["Yellows_Avg"]
        Xone["Red_Diff"] = home_row["Reds_Avg"] - away_row["Reds_Avg"]
        return Xone[self.features] # Ensure column order

    def get_prediction_and_insights(self, Xone: pd.DataFrame, team_avgs: Tuple[pd.DataFrame, pd.DataFrame], home_team: str, away_team: str) -> Tuple[str, List[str]]:
        if not self.is_trained:
            return "Model not trained.", ["Insufficient data."], "N/A"

        pred = self.model.predict(Xone)[0]
        home_stats = team_avgs[0].loc[home_team]
        away_stats = team_avgs[1].loc[away_team]
        
        home_score = home_stats["disc_score"]
        away_score = away_stats["disc_score"]
        
        prediction_text = (
            f"AI Predicts: {home_team} (Home) is the more disciplined team." if pred == 1 
            else f"AI Predicts: {away_team} (Away) is the more disciplined team."
        )
        
        insights = [
            f"Home Team's avg. weighted score is {home_score:.2f} (lower is better).",
            f"Away Team's avg. weighted score is {away_score:.2f} (lower is better).",
        ]
        
        if home_score > away_score and away_stats["Fouls_Avg"] < home_stats["Fouls_Avg"]:
            driver = f"Home's high foul count ({home_stats['Fouls_Avg']:.1f}/game)"
            insights.append(f"This difference is primarily driven by {driver}.")
            summary = f"Exploit {home_team}'s indiscipline by drawing fouls in dangerous areas."
        elif away_score > home_score and home_stats["Fouls_Avg"] < away_stats["Fouls_Avg"]:
            driver = f"Away's high foul count ({away_stats['Fouls_Avg']:.1f}/game)"
            insights.append(f"This difference is primarily driven by {driver}.")
            summary = f"Exploit {away_team}'s indiscipline by drawing fouls in dangerous areas."
        else:
            driver = "yellow card accumulation"
            insights.append(f"The key differentiator appears to be {driver}.")
            summary = "Press the opponent's midfield to force mistakes and potential bookings."

        return prediction_text, insights, summary


class FormationChangeAI:
    """
    AI Model 3: Analyzes "Formation Change".
    Predicts if the match result is likely to change between HT and FT.
    """
    def __init__(self, random_state=RANDOM_STATE):
        self.model = RandomForestClassifier(random_state=random_state, n_estimators=50)
        self.base_features = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "FTHG", "FTAG", "HTHG", "HTAG"]
        self.opt_features = ["HO", "AO"]
        self.features = []
        self.is_trained = False

    def _build_features(self, df: pd.DataFrame, safe_s: callable) -> pd.DataFrame:
        self.features = self.base_features.copy()
        for f in self.opt_features:
            if f in df.columns:
                self.features.append(f)
        
        require_cols(df, self.features)
        X = pd.DataFrame(index=df.index)
        for c in self.features:
            X[c] = safe_s(df[c])
        return X

    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        require_cols(df, ["FTR", "HTR"])
        # 1 = Result changed (e.g., D -> H, H -> D, A -> H), 0 = Result was the same
        y = (df["FTR"] != df["HTR"]).astype(int)
        return y

    def train(self, df: pd.DataFrame, safe_s: callable):
        y = self._build_target(df)
        X = self._build_features(df[y.notna()], safe_s)
        y = y[y.notna()]
        
        if len(X) < 10:
            print("Warning (FormationChangeAI): Not enough data to train.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"FormationChangeAI trained. Accuracy: {self.model.score(X_test, y_test):.2f}")

    def get_team_averages(self, df: pd.DataFrame, safe_s: callable) -> Tuple[pd.DataFrame, pd.DataFrame]:
        home_cols = ["HS", "HST", "HC", "HF", "FTHG", "HTHG"]
        away_cols = ["AS", "AST", "AC", "AF", "FTAG", "HTAG"]
        if "HO" in df.columns: home_cols.append("HO")
        if "AO" in df.columns: away_cols.append("AO")
        
        require_cols(df, home_cols + away_cols)
        
        home_avg = df.groupby("HomeTeam")[home_cols].mean()
        away_avg = df.groupby("AwayTeam")[away_cols].mean()
        
        return home_avg, away_avg

    def build_match_row(self, team_avgs: Tuple[pd.DataFrame, pd.DataFrame], home_team: str, away_team: str) -> Optional[pd.DataFrame]:
        home_avg, away_avg = team_avgs
        try:
            home_row = home_avg.loc[home_team]
            away_row = away_avg.loc[away_team]
        except KeyError:
            print(f"Warning (FormationChangeAI): Missing avg data for {home_team} or {away_team}")
            return None
            
        Xone = pd.DataFrame(index=[0])
        for col in ["HS", "HST", "HC", "HF", "FTHG", "HTHG"]:
            if col in home_row: Xone[col] = home_row[col]
        for col in ["AS", "AST", "AC", "AF", "FTAG", "HTAG"]:
            if col in away_row: Xone[col] = away_row[col]
        
        # Handle optional features
        if "HO" in self.features: Xone["HO"] = home_row.get("HO", 0)
        if "AO" in self.features: Xone["AO"] = away_row.get("AO", 0)
        
        # Ensure all features are present
        for f in self.features:
            if f not in Xone.columns:
                Xone[f] = 0

        return Xone[self.features] # Ensure column order

    def _evaluate_team_side(self, Xone: pd.DataFrame, side: str) -> List[str]:
        """ Rule-based helper for insights. """
        tips = []
        p = "H" if side == "home" else "A"
        o = "A" if side == "home" else "H"
        
        if Xone[f"{p}S"].iloc[0] < 5 and Xone[f"{p}FTHG"].iloc[0] < Xone[f"{o}FTAG"].iloc[0]:
            tips.append("Low shot count ({:.1f}) suggests a need for more attackers.".format(Xone[f'{p}S'].iloc[0]))
        
        if Xone[f"{p}ST"].iloc[0] < max(Xone[f"{p}S"].iloc[0] * 0.2, 1.5):
            tips.append("Poor accuracy ({:.1f} SOT from {:.1f} S) points to finishing issues.".format(Xone[f'{p}ST'].iloc[0], Xone[f'{p}S'].iloc[0]))
        
        if Xone[f"{p}F"].iloc[0] > 15:
            tips.append("High foul count ({:.1f}) may indicate midfield is being overrun.".format(Xone[f'{p}F'].iloc[0]))
        if "HO" in Xone.columns and "AO" in Xone.columns:
             if Xone[f"{p}O"].iloc[0] > 4:
                tips.append("Frequent offsides ({:.1f}) suggest timing/positioning errors.".format(Xone[f'{p}O'].iloc[0]))
        return tips

    def get_prediction_and_insights(self, Xone: pd.DataFrame) -> Tuple[str, List[str]]:
        if not self.is_trained:
            return "Model not trained.", ["Insufficient data."], "N/A"

        pred = self.model.predict(Xone)[0]
        prediction_text = (
            "AI Predicts: Formation Change LIKELY REQUIRED (result may change)." if pred == 1 
            else "AI Predicts: Tactical Stability Expected (result unlikely to change)."
        )
        
        home_tips = self._evaluate_team_side(Xone, "home")
        away_tips = self._evaluate_team_side(Xone, "away")
        
        insights = ["--- Home Team Analysis ---"] + (home_tips if home_tips else ["No obvious tactical flags."]) + \
                   ["--- Away Team Analysis ---"] + (away_tips if away_tips else ["No obvious tactical flags."])
        
        summary = "Both teams show a stable tactical profile based on average performance."
        if pred == 1:
            if home_tips:
                summary = f"Home team must address their {home_tips[0].split('(')[0].lower().strip()}."
            elif away_tips:
                summary = f"Away team must address their {away_tips[0].split('(')[0].lower().strip()}."
            else:
                summary = "A tactical shift is likely; watch for in-game adjustments to exploit."

        return prediction_text, insights, summary


class OpponentWeaknessAI:
    """
    AI Model 4: Analyzes "Target Specific Opponent Weakness".
    Predicts Home Team win based on basic metrics.
    """
    def __init__(self, random_state=RANDOM_STATE):
        self.model = RandomForestClassifier(random_state=random_state, n_estimators=50)
        self.features = ["HS", "AS", "HC", "AC", "HF", "AF"]
        self.is_trained = False

    def _build_features(self, df: pd.DataFrame, safe_s: callable) -> pd.DataFrame:
        require_cols(df, self.features)
        X = pd.DataFrame(index=df.index)
        for c in self.features:
            X[c] = safe_s(df[c])
        return X

    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        require_cols(df, ["FTR"])
        # 1 = Home Win, 0 = Not Home Win (Draw or Away Win)
        y = (df["FTR"] == "H").astype(int)
        return y

    def train(self, df: pd.DataFrame, safe_s: callable):
        y = self._build_target(df)
        X = self._build_features(df[y.notna()], safe_s)
        y = y[y.notna()]
        
        if len(X) < 10:
            print("Warning (OpponentWeaknessAI): Not enough data to train.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"OpponentWeaknessAI trained. Accuracy: {self.model.score(X_test, y_test):.2f}")

    def get_team_averages(self, df: pd.DataFrame, safe_s: callable) -> Tuple[pd.DataFrame, pd.DataFrame]:
        home_cols = ["HS", "HC", "HF"]
        away_cols = ["AS", "AC", "AF"]
        require_cols(df, home_cols + away_cols)
        
        home_avg = df.groupby("HomeTeam")[home_cols].mean()
        away_avg = df.groupby("AwayTeam")[away_cols].mean()
        
        return home_avg, away_avg

    def build_match_row(self, team_avgs: Tuple[pd.DataFrame, pd.DataFrame], home_team: str, away_team: str) -> Optional[pd.DataFrame]:
        home_avg, away_avg = team_avgs
        try:
            home_row = home_avg.loc[home_team]
            away_row = away_avg.loc[away_team]
        except KeyError:
            print(f"Warning (OpponentWeaknessAI): Missing avg data for {home_team} or {away_team}")
            return None
            
        Xone = pd.DataFrame(index=[0])
        # Assign in the order of self.features
        Xone["HS"] = home_row["HS"]
        Xone["AS"] = away_row["AS"]
        Xone["HC"] = home_row["HC"]
        Xone["AC"] = away_row["AC"]
        Xone["HF"] = home_row["HF"]
        Xone["AF"] = away_row["AF"]
        
        return Xone[self.features] # Ensure column order

    def _weakness_insights(self, Xone: pd.DataFrame, side: str) -> List[str]:
        tips = []
        p = "H" if side == "home" else "A"
        o = "A" if side == "home" else "H"
        
        if Xone[f"{o}F"].iloc[0] > Xone[f"{p}F"].iloc[0] * 1.2:
            tips.append(f"Exploit Opponent: High foul rate ({Xone[f'{o}F'].iloc[0]:.1f}) suggests vulnerability to direct runners.")
        if Xone[f"{o}C"].iloc[0] < Xone[f"{p}C"].iloc[0] * 0.8:
            tips.append(f"Exploit Opponent: Low corner count ({Xone[f'{o}C'].iloc[0]:.1f}) implies weak offensive pressure.")
        if Xone[f"{o}S"].iloc[0] < Xone[f"{p}S"].iloc[0] * 0.8:
             tips.append(f"Exploit Opponent: Low shot volume ({Xone[f'{o}S'].iloc[0]:.1f}) allows for a higher defensive line.")
        
        if not tips:
            tips.append("Opponent shows no obvious statistical weaknesses in these metrics.")
        return tips

    def get_prediction_and_insights(self, Xone: pd.DataFrame, home_team: str, away_team: str) -> Tuple[str, List[str]]:
        if not self.is_trained:
            return "Model not trained.", ["Insufficient data."], "N/A"
            
        # Get class index for "1" (Home Win)
        try:
            class_index = np.where(self.model.classes_ == 1)[0][0]
            prob = self.model.predict_proba(Xone)[0][class_index]
            prediction_text = f"AI Predicts Home ({home_team}) Win Probability: {prob:.0%}"
        except (IndexError, NotFittedError, ValueError):
            # Model might not have seen "1" if data is skewed (unlikely) or not fitted
            prediction_text = "AI prediction unavailable (model data issue)."
            prob = 0.5
            
        if prob > 0.6:
            insights = self._weakness_insights(Xone, "home")
            summary = f"Focus attacks based on {insights[0].split(':')[1].split('(')[0].strip()}."
        elif prob < 0.4:
            insights = self._weakness_insights(Xone, "away")
            summary = f"Home team must be wary; Away team can exploit {insights[0].split('(')[0].strip()}."
        else:
            insights = ["Match is evenly balanced. No clear statistical weakness identified."]
            summary = "This will be a tight game. Look for individual errors to make the difference."

        return prediction_text, insights, summary


class SubstitutionAI:
    """
    AI Model 5: Analyzes "Substitutions".
    Predicts if a team will have a "Good" performance (>0 score) or "Bad" (<=0).
    """
    def __init__(self, random_state=RANDOM_STATE):
        self.model = RandomForestClassifier(random_state=random_state, n_estimators=50)
        self.perf_cols = ["HS", "HST", "HF", "HC", "AS", "AST", "AF", "AC", "FTHG", "FTAG"]
        self.features = []
        self.is_trained = False

    def _build_performance_score(self, df: pd.DataFrame, safe_s: callable, side: str) -> pd.Series:
        """ Calculates a simple performance score. """
        if side == "home":
            score = (
                (safe_s(df["FTHG"]) - safe_s(df["FTAG"])) * 3 + # Win/Loss
                safe_s(df["HST"]) * 0.5 +                     # Shots on Target
                safe_s(df["HS"]) * 0.2 +                      # Shots
                safe_s(df["HC"]) * 0.1 -                      # Corners
                safe_s(df["HF"]) * 0.2                        # Fouls
            )
        else: # away
             score = (
                (safe_s(df["FTAG"]) - safe_s(df["FTHG"])) * 3 +
                safe_s(df["AST"]) * 0.5 +
                safe_s(df["AS"]) * 0.2 +
                safe_s(df["AC"]) * 0.1 -
                safe_s(df["AF"]) * 0.2
            )
        return score

    def _build_features(self, df: pd.DataFrame, safe_s: callable) -> pd.DataFrame:
        self.features = [f"{p}_{t}" for t in ["Home", "Away"] for p in ["Shots", "OnTarget", "Fouls", "Corners"]]
        
        X = pd.DataFrame(index=df.index)
        X["Shots_Home"] = safe_s(df["HS"])
        X["OnTarget_Home"] = safe_s(df["HST"])
        X["Fouls_Home"] = safe_s(df["HF"])
        X["Corners_Home"] = safe_s(df["HC"])
        X["Shots_Away"] = safe_s(df["AS"])
        X["OnTarget_Away"] = safe_s(df["AST"])
        X["Fouls_Away"] = safe_s(df["AF"])
        X["Corners_Away"] = safe_s(df["AC"])
        return X

    def _build_target(self, df: pd.DataFrame, safe_s: callable) -> Tuple[pd.Series, pd.Series]:
        # This model is different: it analyzes team-by-team, not match-by-match
        home_df = df.copy()
        home_df["Team"] = home_df["HomeTeam"]
        home_df["Opponent"] = home_df["AwayTeam"]
        home_df["Score"] = self._build_performance_score(home_df, safe_s, "home")
        
        away_df = df.copy()
        away_df["Team"] = away_df["AwayTeam"]
        away_df["Opponent"] = away_df["HomeTeam"]
        away_df["Score"] = self._build_performance_score(away_df, safe_s, "away")
        
        # We need features relative to the team
        home_X = pd.DataFrame(index=home_df.index)
        home_X["Shots"] = safe_s(home_df["HS"])
        home_X["OnTarget"] = safe_s(home_df["HST"])
        home_X["Fouls"] = safe_s(home_df["HF"])
        home_X["Corners"] = safe_s(home_df["HC"])
        home_X["Opp_Shots"] = safe_s(home_df["AS"])
        home_X["Opp_OnTarget"] = safe_s(home_df["AST"])
        home_X["Opp_Fouls"] = safe_s(home_df["AF"])
        home_X["Opp_Corners"] = safe_s(home_df["AC"])
        home_y = (home_df["Score"] > 0).astype(int) # 1 = Good, 0 = Bad

        away_X = pd.DataFrame(index=away_df.index)
        away_X["Shots"] = safe_s(away_df["AS"])
        away_X["OnTarget"] = safe_s(away_df["AST"])
        away_X["Fouls"] = safe_s(away_df["AF"])
        away_X["Corners"] = safe_s(away_df["AC"])
        away_X["Opp_Shots"] = safe_s(away_df["HS"])
        away_X["Opp_OnTarget"] = safe_s(away_df["HST"])
        away_X["Opp_Fouls"] = safe_s(away_df["HF"])
        away_X["Opp_Corners"] = safe_s(away_df["HC"])
        away_y = (away_df["Score"] > 0).astype(int)
        
        self.features = list(home_X.columns)
        
        # Combine home and away perspectives
        X_combined = pd.concat([home_X, away_X], ignore_index=True)
        y_combined = pd.concat([home_y, away_y], ignore_index=True)
        
        return X_combined, y_combined

    def train(self, df: pd.DataFrame, safe_s: callable):
        X, y = self._build_target(df, safe_s)
        
        if len(X) < 10:
            print("Warning (SubstitutionAI): Not enough data to train.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"SubstitutionAI trained. Accuracy: {self.model.score(X_test, y_test):.2f}")

    def get_team_averages(self, df: pd.DataFrame, safe_s: callable) -> Dict[str, float]:
        """
        Returns a dictionary of {team_name: avg_performance_score}.
        """
        home_scores = self._build_performance_score(df, safe_s, "home")
        away_scores = self._build_performance_score(df, safe_s, "away")
        
        home_perf = pd.DataFrame({"Team": df["HomeTeam"], "Score": home_scores})
        away_perf = pd.DataFrame({"Team": df["AwayTeam"], "Score": away_scores})
        
        all_perf = pd.concat([home_perf, away_perf])
        team_avg_scores = all_perf.groupby("Team")["Score"].mean()
        
        return team_avg_scores.to_dict()

    def build_match_row(self, team_avgs: Any, home_team: str, away_team: str, all_team_stats: Tuple[pd.DataFrame, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Builds the feature row for prediction.
        'team_avgs' for this model is just the score dict, so we need other stats.
        """
        home_avg_stats, away_avg_stats = all_team_stats
        try:
            home_row = home_avg_stats.loc[home_team]
            away_row = away_avg_stats.loc[away_team]
        except KeyError:
            print(f"Warning (SubstitutionAI): Missing avg data for {home_team} or {away_team}")
            return None

        # Build the feature row from the perspective of the *Home Team*
        Xone_home = pd.DataFrame(index=[0])
        Xone_home["Shots"] = home_row.get("HS", 0)
        Xone_home["OnTarget"] = home_row.get("HST", 0)
        Xone_home["Fouls"] = home_row.get("HF", 0)
        Xone_home["Corners"] = home_row.get("HC", 0)
        Xone_home["Opp_Shots"] = away_row.get("AS", 0)
        Xone_home["Opp_OnTarget"] = away_row.get("AST", 0)
        Xone_home["Opp_Fouls"] = away_row.get("AF", 0)
        Xone_home["Opp_Corners"] = away_row.get("AC", 0)
        
        # Build the feature row from the perspective of the *Away Team*
        Xone_away = pd.DataFrame(index=[0])
        Xone_away["Shots"] = away_row.get("AS", 0)
        Xone_away["OnTarget"] = away_row.get("AST", 0)
        Xone_away["Fouls"] = away_row.get("AF", 0)
        Xone_away["Corners"] = away_row.get("AC", 0)
        Xone_away["Opp_Shots"] = home_row.get("HS", 0)
        Xone_away["Opp_OnTarget"] = home_row.get("HST", 0)
        Xone_away["Opp_Fouls"] = home_row.get("HF", 0)
        Xone_away["Opp_Corners"] = home_row.get("HC", 0)

        return Xone_home[self.features], Xone_away[self.features] # Return both perspectives

    def get_prediction_and_insights(self, Xones: Tuple[pd.DataFrame, pd.DataFrame], team_avg_scores: Dict[str, float], home_team: str, away_team: str) -> Tuple[str, List[str]]:
        if not self.is_trained:
            return "Model not trained.", ["Insufficient data."], "N/A"

        Xone_home, Xone_away = Xones
        
        pred_home = self.model.predict(Xone_home)[0]
        pred_away = self.model.predict(Xone_away)[0]
        
        home_score = team_avg_scores.get(home_team, 0)
        away_score = team_avg_scores.get(away_team, 0)
        
        prediction_text = (
            f"AI predicts a 'Good' performance ({pred_home}) for Home and 'Good' ({pred_away}) for Away."
            if pred_home == 1 and pred_away == 1 else
            f"AI predicts a 'Good' performance ({pred_home}) for Home and 'Bad' ({pred_away}) for Away."
            if pred_home == 1 else
            f"AI predicts a 'Bad' performance ({pred_home}) for Home and 'Good' ({pred_away}) for Away."
            if pred_away == 1 else
            f"AI predicts a 'Bad' performance ({pred_home}) for Home and 'Bad' ({pred_away}) for Away."
        )
        
        insights = [
            f"{home_team} (Home) shows a higher average performance score ({home_score:.2f}).",
            f"{away_team} (Away) shows a lower average performance score ({away_score:.2f})."
        ] if home_score > away_score else [
            f"{away_team} (Away) shows a higher average performance score ({away_score:.2f}).",
            f"{home_team} (Home) shows a lower average performance score ({home_score:.2f})."
        ]
        
        if home_score < 0 or away_score < 0:
            insights.append("One or both teams average a negative performance score, indicating consistent underperformance.")
            
        summary = "Prepare to make proactive changes around the 60-minute mark to exploit fatigue or poor performance."
        if home_score > (away_score + 1.0):
            summary = f"{home_team}'s superior performance metrics suggest they can outlast {away_team}. Look for {away_team} to tire first."
        elif away_score > (home_score + 1.0):
             summary = f"{away_team}'s superior performance metrics suggest they can outlast {home_team}. Look for {home_team} to tire first."

        return prediction_text, insights, summary


# ----------------------------------------------------------------------
# 4. CHART GENERATION
# ----------------------------------------------------------------------

def setup_chart_style():
    """ Sets a consistent style for all charts. """
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_attack_defend(home_stats: Dict, away_stats: Dict, path: str):
    """ Grouped Bar Chart: Offensive Output vs. Defensive Infractions. """
    setup_chart_style()
    data = {
        "Team": ["Home", "Home", "Away", "Away"],
        "Metric": ["Offensive Output", "Defensive Infractions", "Offensive Output", "Defensive Infractions"],
        "Value": [
            home_stats["Offensive"],
            home_stats["Defensive"],
            away_stats["Offensive"],
            away_stats["Defensive"]
        ]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="Metric", y="Value", hue="Team", palette=["#1f77b4", "#ff7f0e"])
    ax.set_title("Offensive Output vs. Defensive Infractions", weight="bold")
    ax.set_ylabel("Average per Game")
    ax.set_xlabel("")
    plt.legend(title="Team")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_discipline(home_stats: Dict, away_stats: Dict, path: str):
    """ Horizontal Stacked Bar Chart: Disciplinary Risk Profile. """
    setup_chart_style()
    data = {
        "Team": ["Home", "Away"],
        "Fouls (x1)": [home_stats["Fouls"], away_stats["Fouls"]],
        "Yellows (x5)": [home_stats["Yellows"], away_stats["Yellows"]],
        "Reds (x15)": [home_stats["Reds"], away_stats["Reds"]],
    }
    df = pd.DataFrame(data).set_index("Team")
    
    ax = df.plot(kind='barh', stacked=True, figsize=(10, 6), colormap="autumn")
    ax.set_title("Disciplinary Risk Profile (Weighted Score)", weight="bold")
    ax.set_xlabel("Weighted Score (Lower is Better)")
    ax.set_ylabel("Team")
    
    # Add total labels
    df_total = df.sum(axis=1)
    for i, total in enumerate(df_total):
        ax.text(total + 0.5, i, f"Total: {total:.1f}", va='center', weight='bold')
        
    plt.legend(title="Infraction (Weighted)", loc='lower right')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_formation_radar(home_stats: List, away_stats: List, path: str):
    """ Radar Chart: Tactical Profile Comparison. """
    setup_chart_style()
    categories = ['Shots', 'On Target', 'Corners', 'Fouls', 'Offsides']
    N = len(categories)
    
    # Normalize data for radar chart (0-100 scale)
    # Find max value for each category across both teams to set the scale
    max_vals = [max(h, a) if max(h, a) > 0 else 1 for h, a in zip(home_stats, away_stats)]
    home_norm = [(x / max_v) * 100 for x, max_v in zip(home_stats, max_vals)]
    away_norm = [(x / max_v) * 100 for x, max_v in zip(away_stats, max_vals)]

    # Angles for radar chart
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    home_norm += home_norm[:1]
    away_norm += away_norm[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=8)
    plt.ylim(0, 100)

    # Plot Home
    ax.plot(angles, home_norm, linewidth=2, linestyle='solid', label='Home', color="#1f77b4")
    ax.fill(angles, home_norm, alpha=0.1, color="#1f77b4")
    
    # Plot Away
    ax.plot(angles, away_norm, linewidth=2, linestyle='solid', label='Away', color="#ff7f0e")
    ax.fill(angles, away_norm, alpha=0.1, color="#ff7f0e")
    
    plt.title("Tactical Profile Comparison (Normalized)", size=14, weight="bold", y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(path)
    plt.close()

def plot_opponent_weakness(home_stats: List, away_stats: List, path: str):
    """ Grouped Bar Chart: Key Battleground Metrics. """
    setup_chart_style()
    data = {
        "Metric": ["Shots", "Corners", "Fouls", "Shots", "Corners", "Fouls"],
        "Team": ["Home", "Home", "Home", "Away", "Away", "Away"],
        "Value": home_stats + away_stats
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="Metric", y="Value", hue="Team", palette=["#1f77b4", "#ff7f0e"])
    ax.set_title("Key Battleground Metrics", weight="bold")
    ax.set_ylabel("Average per Game")
    ax.set_xlabel("")
    plt.legend(title="Team")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
def plot_substitution(home_score: float, away_score: float, path: str):
    """ Bar Chart: Average Match Performance Score. """
    setup_chart_style()
    data = {"Team": ["Home", "Away"], "Score": [home_score, away_score]}
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(data=df, x="Team", y="Score", palette=["#1f77b4", "#ff7f0e"])
    ax.set_title("Average Match Performance Score", weight="bold")
    ax.set_ylabel("Avg. Performance Score")
    ax.set_xlabel("")
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_set_piece(home_opps: float, away_opps: float, path: str):
    """ Donut Chart: Set Piece Dominance. """
    setup_chart_style()
    labels = [f"Home ({home_opps:.1f})", f"Away ({away_opps:.1f})"]
    sizes = [home_opps, away_opps]
    colors = ["#1f77b4", "#ff7f0e"]
    
    if home_opps <= 0 and away_opps <= 0:
        sizes = [1, 1] # Avoid division by zero
        
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=colors, pctdistance=0.85,
           textprops={'color': '#333333', 'weight': 'bold'})
    
    # Draw circle for donut
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    
    ax.axis('equal')  # Equal aspect ratio
    plt.title("Set Piece Dominance\n(Avg. Corners Won + Fouls Drawn)", weight="bold")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ----------------------------------------------------------------------
# 5. DATA ORCHESTRATION & MAIN APPLICATION
# ----------------------------------------------------------------------

def load_and_prep_data(csv_path: str) -> pd.DataFrame:
    """
    Loads and applies initial preparation to the main dataset.
    """
    print(f"Loading data from {csv_file_path}...")
    df = load_df(csv_file_path)
    df = compute_booking_points_if_missing(df)
    print(f"Data loaded. {len(df)} matches found.")
    return df

def train_all_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Initializes all 5 AI classes and trains them.
    """
    print("\n--- Training AI Models ---")
    ai_models = {
        "attack_defend": AttackDefendAI(),
        "discipline": DisciplineAI(),
        "formation_change": FormationChangeAI(),
        "opponent_weakness": OpponentWeaknessAI(),
        "substitution": SubstitutionAI(),
    }
    
    ai_models["attack_defend"].train(df, safe_series)
    ai_models["discipline"].train(df, safe_series)
    ai_models["formation_change"].train(df, safe_series)
    ai_models["opponent_weakness"].train(df, safe_series)
    ai_models["substitution"].train(df, safe_series)
    
    print("--- AI Model Training Complete ---")
    return ai_models

def get_user_input(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Prompts the user for Home and Away teams with autocompletion.
    """
    teams = sorted(list(set(df["HomeTeam"]).union(df["AwayTeam"])))
    completer = WordCompleter(teams, ignore_case=True)
    
    print("\n--- Team Selection ---")
    print(f"Found {len(teams)} teams. Examples: {', '.join(teams[:5])}...")
    
    while True:
        home_team = prompt("Enter HOME team: ", completer=completer).strip()
        if home_team in teams:
            break
        print(f"Invalid team: '{home_team}'. Please select from the list.")
        
    while True:
        away_team = prompt("Enter AWAY team: ", completer=completer).strip()
        if away_team in teams:
            if away_team == home_team:
                print("Away team cannot be the same as the Home team.")
            else:
                break
        print(f"Invalid team: '{away_team}'. Please select from the list.")
        
    return home_team, away_team

def get_generic_team_averages(df: pd.DataFrame, safe_s: callable) -> Dict[str, pd.DataFrame]:
    """
    Calculates generic averages needed for charts and new evaluators.
    """
    # Base stats for home/away
    home_cols = ["HS", "HST", "HC", "HF", "HY", "HR", "FTHG", "HTHG"]
    away_cols = ["AS", "AST", "AC", "AF", "AY", "AR", "FTAG", "HTAG"]
    if "HO" in df.columns: home_cols.append("HO")
    if "AO" in df.columns: away_cols.append("AO")
    require_cols(df, home_cols + away_cols)
    
    home_avg = df.groupby("HomeTeam")[home_cols].mean()
    away_avg = df.groupby("AwayTeam")[away_cols].mean()
    
    # Specific stats for Set Piece (Fouls Drawn)
    home_fouls_drawn = df.groupby("HomeTeam")["AF"].mean().rename("Fouls_Drawn_Home")
    away_fouls_drawn = df.groupby("AwayTeam")["HF"].mean().rename("Fouls_Drawn_Away")
    
    all_teams = sorted(list(set(df["HomeTeam"]).union(df["AwayTeam"])))
    stats_dict = {}
    for team in all_teams:
        stats_dict[team] = {
            "Avg_HC_Won": home_avg.loc[team]["HC"] if team in home_avg.index else 0,
            "Avg_AC_Won": away_avg.loc[team]["AC"] if team in away_avg.index else 0,
            "Avg_HF_Drawn": home_fouls_drawn.loc[team] if team in home_fouls_drawn.index else 0,
            "Avg_AF_Drawn": away_fouls_drawn.loc[team] if team in away_fouls_drawn.index else 0,
        }
        
    return {"home": home_avg, "away": away_avg, "set_piece": stats_dict}


def generate_report_data(
    ai_models: Dict[str, Any], 
    df: pd.DataFrame, 
    home_team: str, 
    away_team: str
) -> Dict[str, Any]:
    """
    Runs all analyses, generates charts, and collects data for the PDF.
    """
    print("\n--- Generating Report Data ---")
    os.makedirs(CHART_DIR, exist_ok=True)
    report_content = {}
    
    # Get generic stats once
    generic_stats = get_generic_team_averages(df, safe_series)
    home_avg_basic = generic_stats["home"]
    away_avg_basic = generic_stats["away"]
    
    # 1. Attack/Defend
    try:
        print("Analyzing: Attack/Defend")
        handler = ai_models["attack_defend"]
        avgs = handler.get_team_averages(df, safe_series)
        Xone = handler.build_match_row(avgs, home_team, away_team)
        pred, insights, summary = handler.get_prediction_and_insights(Xone)
        
        home_s = home_avg_basic.loc[home_team]
        away_s = away_avg_basic.loc[away_team]
        chart_path = os.path.join(CHART_DIR, "attack_defend_chart.png")
        plot_attack_defend(
            {"Offensive": home_s["HS"] + home_s["HST"], "Defensive": home_s["HF"] + home_s["HY"]},
            {"Offensive": away_s["AS"] + away_s["AST"], "Defensive": away_s["AF"] + away_s["AY"]},
            chart_path
        )
        report_content["attack_defend"] = {
            "pred": pred, "insights": insights, "summary": summary, "chart_path": chart_path
        }
    except Exception as e:
        print(f"Error in Attack/Defend AI: {e}")
        report_content["attack_defend"] = None

    # 2. Discipline
    try:
        print("Analyzing: Discipline")
        handler = ai_models["discipline"]
        avgs = handler.get_team_averages(df, safe_series)
        Xone = handler.build_match_row(avgs, home_team, away_team)
        pred, insights, summary = handler.get_prediction_and_insights(Xone, avgs, home_team, away_team)
        
        home_s = avgs[0].loc[home_team]
        away_s = avgs[1].loc[away_team]
        chart_path = os.path.join(CHART_DIR, "discipline_chart.png")
        plot_discipline(
            {"Fouls": home_s["Fouls_Avg"] * handler.WEIGHTS["foul_point"], 
             "Yellows": home_s["Yellows_Avg"] * handler.WEIGHTS["yellow_point"], 
             "Reds": home_s["Reds_Avg"] * handler.WEIGHTS["red_point"]},
            {"Fouls": away_s["Fouls_Avg"] * handler.WEIGHTS["foul_point"], 
             "Yellows": away_s["Yellows_Avg"] * handler.WEIGHTS["yellow_point"], 
             "Reds": away_s["Reds_Avg"] * handler.WEIGHTS["red_point"]},
            chart_path
        )
        report_content["discipline"] = {
            "pred": pred, "insights": insights, "summary": summary, "chart_path": chart_path
        }
    except Exception as e:
        print(f"Error in Discipline AI: {e}")
        report_content["discipline"] = None

    # 3. Formation Change
    try:
        print("Analyzing: Formation Change")
        handler = ai_models["formation_change"]
        avgs = handler.get_team_averages(df, safe_series)
        Xone = handler.build_match_row(avgs, home_team, away_team)
        pred, insights, summary = handler.get_prediction_and_insights(Xone)
        
        home_s = home_avg_basic.loc[home_team]
        away_s = away_avg_basic.loc[away_team]
        chart_path = os.path.join(CHART_DIR, "formation_chart.png")
        plot_formation_radar(
            [home_s.get("HS", 0), home_s.get("HST", 0), home_s.get("HC", 0), home_s.get("HF", 0), home_s.get("HO", 0)],
            [away_s.get("AS", 0), away_s.get("AST", 0), away_s.get("AC", 0), away_s.get("AF", 0), away_s.get("AO", 0)],
            chart_path
        )
        report_content["formation_change"] = {
            "pred": pred, "insights": insights, "summary": summary, "chart_path": chart_path
        }
    except Exception as e:
        print(f"Error in Formation Change AI: {e}")
        report_content["formation_change"] = None

    # 4. Opponent Weakness
    try:
        print("Analyzing: Opponent Weakness")
        handler = ai_models["opponent_weakness"]
        avgs = handler.get_team_averages(df, safe_series)
        Xone = handler.build_match_row(avgs, home_team, away_team)
        pred, insights, summary = handler.get_prediction_and_insights(Xone, home_team, away_team)
        
        home_s = avgs[0].loc[home_team]
        away_s = avgs[1].loc[away_team]
        chart_path = os.path.join(CHART_DIR, "opponent_weakness_chart.png")
        plot_opponent_weakness(
            [home_s["HS"], home_s["HC"], home_s["HF"]],
            [away_s["AS"], away_s["AC"], away_s["AF"]],
            chart_path
        )
        report_content["opponent_weakness"] = {
            "pred": pred, "insights": insights, "summary": summary, "chart_path": chart_path
        }
    except Exception as e:
        print(f"Error in Opponent Weakness AI: {e}")
        report_content["opponent_weakness"] = None

    # 5. Substitution
    try:
        print("Analyzing: Substitution")
        handler = ai_models["substitution"]
        team_avg_scores = handler.get_team_averages(df, safe_series)
        Xones = handler.build_match_row(team_avg_scores, home_team, away_team, (home_avg_basic, away_avg_basic))
        pred, insights, summary = handler.get_prediction_and_insights(Xones, team_avg_scores, home_team, away_team)
        
        home_score = team_avg_scores.get(home_team, 0)
        away_score = team_avg_scores.get(away_team, 0)
        chart_path = os.path.join(CHART_DIR, "substitution_chart.png")
        plot_substitution(home_score, away_score, chart_path)
        
        report_content["substitution"] = {
            "pred": pred, "insights": insights, "summary": summary, "chart_path": chart_path
        }
    except Exception as e:
        print(f"Error in Substitution AI: {e}")
        report_content["substitution"] = None

    # 6. Set Piece (New)
    try:
        print("Analyzing: Set Piece")
        sp_stats = generic_stats["set_piece"]
        home_opps = sp_stats[home_team]["Avg_HC_Won"] + sp_stats[home_team]["Avg_AF_Drawn"]
        away_opps = sp_stats[away_team]["Avg_AC_Won"] + sp_stats[away_team]["Avg_HF_Drawn"]
        
        chart_path = os.path.join(CHART_DIR, "set_piece_chart.png")
        plot_set_piece(home_opps, away_opps, chart_path)
        
        insights = [
            f"{home_team} (Home) averages {sp_stats[home_team]['Avg_HC_Won']:.1f} corners won per game.",
            f"{home_team} (Home) draws an average of {sp_stats[home_team]['Avg_AF_Drawn']:.1f} fouls from opponents when away (proxy for fouls drawn).",
            f"{away_team} (Away) averages {sp_stats[away_team]['Avg_AC_Won']:.1f} corners won per game.",
            f"{away_team} (Away) draws an average of {sp_stats[away_team]['Avg_HF_Drawn']:.1f} fouls from opponents when home (proxy for fouls drawn)."
        ]
        
        summary = "This is a key opportunity. Prioritize aerial threats and set-piece routines."
        if home_opps > away_opps * 1.2:
            summary = f"{home_team} has a clear advantage in set piece opportunities. This is a key route to goal."
        elif away_opps > home_opps * 1.2:
            summary = f"{away_team} generates more set piece situations. {home_team} must be disciplined in defense."
        
        report_content["set_piece"] = {
            "insights": insights, "summary": summary, "chart_path": chart_path
        }
    except Exception as e:
        print(f"Error in Set Piece Analysis: {e}")
        report_content["set_piece"] = None
        
    print("--- Analysis Complete ---")
    return report_content


# ----------------------------------------------------------------------
# 6. PDF GENERATION (using fpdf2)
# ----------------------------------------------------------------------

class PDFReport(FPDF):
    """
    Custom PDF class to create a neat report with headers and footers.
    """
    def __init__(self, home_team, away_team):
        super().__init__()
        self.home_team = home_team
        self.away_team = away_team
        self.set_auto_page_break(auto=True, margin=15)
        self.set_left_margin(15)
        self.set_right_margin(15)
        # Use core fonts
        
    def header(self):
        if self.page_no() == 1:
            return  # No header on title page
        self.set_font("Arial", "", 9)
        # *** FIX HERE: Replaced ln=0 with new_x/new_y to fix warnings ***
        self.cell(0, 10, f"Scout Report: {self.home_team} vs {self.away_team}", 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='L')
        self.cell(0, 10, f"Page {self.page_no()}", 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='R')
        self.ln(15)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Confidential AI Report | Generated: {datetime.date.today().isoformat()}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    def add_title_page(self):
        self.add_page()
        self.set_font("Arial", "B", 28)
        self.cell(0, 30, "AI-Powered Scout Report", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)
        
        self.set_font("Arial", "B", 22)
        self.cell(0, 20, self.home_team, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font("Arial", "", 18)
        self.cell(0, 15, "vs", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font("Arial", "B", 22)
        self.cell(0, 20, self.away_team, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(20)
        
        self.set_font("Arial", "", 12)
        self.cell(0, 10, f"Date Generated: {datetime.date.today().isoformat()}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(20)

    def add_executive_summary(self, content: Dict[str, Any]):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Executive Summary", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)
        
        self.set_font("Arial", "", 11)
        
        summary_points = []
        try:
            if content.get("attack_defend"):
                summary_points.append(f"- Attack/Defend AI: {content['attack_defend']['pred']}")
            if content.get("discipline"):
                summary_points.append(f"- Discipline AI: {content['discipline']['pred']}")
            if content.get("formation_change"):
                summary_points.append(f"- Tactical AI: {content['formation_change']['pred']}")
            if content.get("opponent_weakness"):
                summary_points.append(f"- Weakness AI: {content['opponent_weakness']['pred']}")
            if content.get("set_piece"):
                summary_points.append(f"- Key Insight: {content['set_piece']['summary']}")
            
            if not summary_points:
                 summary_points = ["- Report generation incomplete due to data errors."]

        except Exception as e:
            print(f"Error building summary: {e}")
            summary_points = ["- Report generation incomplete due to data errors."]
            
        for point in summary_points:
            self.multi_cell(0, 6, point, border=0, align='L')
            self.ln(2)

    # *** FIX HERE: Complete re-write of this function's layout logic ***
    def add_ai_page(self, title: str, chart_path: str, pred_text: Optional[str], insights: List[str], summary: str):
        self.add_page()
        
        # Page Title
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, title, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)
        
        # Two-column layout: Chart on left, Text on right
        y_start = self.get_y()
        
        # Chart (left)
        if chart_path and os.path.exists(chart_path):
            self.image(chart_path, x=15, y=y_start, w=100)
        else:
            self.set_xy(15, y_start)
            self.set_font("Arial", "I", 10)
            self.multi_cell(100, 10, "Chart could not be generated.", 1, align='C')

        # --- Text (right) ---
        # Set start position for the right column
        self.set_xy(125, y_start) 
        # Calculate the width of the right column
        col_width = self.w - self.r_margin - 125 # 210 - 15 - 125 = 70 mm
        
        if pred_text:
            self.set_font("Arial", "B", 12)
            self.cell(col_width, 7, "AI Decision:", 0, new_y=YPos.NEXT, align='L')
            self.set_font("Arial", "", 11)
            # Use col_width, not 0, to constrain the text
            self.multi_cell(col_width, 5, pred_text, border=0, align='L') 
            self.ln(4)
        
        # Reset X to the right column after self.ln()
        self.set_x(125) 
        self.set_font("Arial", "B", 12)
        self.cell(col_width, 7, "AI Insights:", 0, new_y=YPos.NEXT, align='L')
        self.set_font("Arial", "", 10)
        for insight in insights:
            # Reset X for *every* multi_cell to keep it in the column
            self.set_x(125) 
            self.multi_cell(col_width, 5, f"- {insight}", border=0, align='L')
        self.ln(4)
        
        # Reset X to the right column after self.ln()
        self.set_x(125) 
        self.set_font("Arial", "B", 12)
        self.cell(col_width, 7, "Managerial Bottom Line:", 0, new_y=YPos.NEXT, align='L')
        self.set_font("Arial", "I", 11)
        # Reset X one last time for the final block
        self.set_x(125)
        self.multi_cell(col_width, 5, summary, border=0, align='L')

        # Move below the taller of the two columns (chart is approx 80mm high)
        self.set_y(max(y_start + 85, self.get_y() + 10))


def create_report(report_content: Dict[str, Any], home_team: str, away_team: str, filename: str):
    """
    Assembles the final PDF document.
    """
    print(f"\n--- Creating PDF Report: {filename} ---")
    pdf = PDFReport(home_team, away_team)
    
    # Page 1: Title & Summary
    pdf.add_title_page()
    pdf.add_executive_summary(report_content)
    
    # Page 2: Attack/Defend
    if report_content.get("attack_defend"):
        data = report_content["attack_defend"]
        pdf.add_ai_page(
            "Performance Evaluator: Attack vs. Defense Strategy",
            data["chart_path"], data["pred"], data["insights"], data["summary"]
        )

    # Page 3: Discipline
    if report_content.get("discipline"):
        data = report_content["discipline"]
        pdf.add_ai_page(
            "Performance Evaluator: Disciplinary Profile",
            data["chart_path"], data["pred"], data["insights"], data["summary"]
        )

    # Page 4: Formation
    if report_content.get("formation_change"):
        data = report_content["formation_change"]
        pdf.add_ai_page(
            "Performance Evaluator: Formation & Tactical Shape",
            data["chart_path"], data["pred"], data["insights"], data["summary"]
        )

    # Page 5: Opponent Weakness
    if report_content.get("opponent_weakness"):
        data = report_content["opponent_weakness"]
        pdf.add_ai_page(
            "Performance Evaluator: Opponent Weakness",
            data["chart_path"], data["pred"], data["insights"], data["summary"]
        )

    # Page 6: Substitution
    if report_content.get("substitution"):
        data = report_content["substitution"]
        pdf.add_ai_page(
            "Performance Evaluator: Substitution Decisions",
            data["chart_path"], data["pred"], data["insights"], data["summary"]
        )

    # Page 7: Set Piece
    if report_content.get("set_piece"):
        data = report_content["set_piece"]
        pdf.add_ai_page(
            "Performance Evaluator: Set Piece Opportunities",
            data["chart_path"], None, data["insights"], data["summary"]
        )

    pdf.output(filename)
    print(f"--- PDF Report Saved ---")


# ----------------------------------------------------------------------
# 7. MAIN EXECUTION BLOCK
# ----------------------------------------------------------------------

# Global variable to hold the CSV path
csv_file_path = ""

def main():
    """
    Main function to run the entire pipeline.
    """
    global csv_file_path # Use the global variable
    
    # Try to find the data file in a few common locations
    possible_paths = [
        "E0.csv",
        "C:\\Users\\Prasiddha\\OneDrive\\Desktop\\Capstone\\Feature selection\\E0.csv",
        os.path.join(os.path.dirname(__file__), "E0.csv"),
    ]
    
    DATA_PATH = None
    for path in possible_paths:
        if os.path.isfile(path):
            DATA_PATH = path
            break
            
    if DATA_PATH is None:
        print(f"Error: Data file 'E0.csv' not found.")
        print("Please ensure 'E0.csv' is in one of these locations:")
        for path in possible_paths:
            print(f"- {os.path.abspath(path)}")
        sys.exit(1)
    
    csv_file_path = DATA_PATH # Set the global path
        
    try:
        # 1. Load Data
        df = load_and_prep_data(DATA_PATH)
        
        # 2. Train Models
        ai_models = train_all_models(df)
        
        # 3. Get User Input
        home_team, away_team = get_user_input(df)
        
        # 4. Generate Analysis & Charts
        report_data = generate_report_data(ai_models, df, home_team, away_team)
        
        # 5. Create PDF
        output_filename = f"Scout_Report_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}.pdf"
        create_report(report_data, home_team, away_team, output_filename)
        
        print(f"\n✅ Success! Report generated: {output_filename}")
        print(f"Chart files saved in: '{CHART_DIR}' directory.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()