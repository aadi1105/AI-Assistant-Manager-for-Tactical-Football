#!/usr/bin/env python3

"""
Football Scouting Report Generator
A comprehensive ML-powered scouting system for Premier League teams
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# METRIC CATEGORIES CONFIGURATION
# ============================================================================

METRIC_CATEGORIES = {
    "1. Predicted Lineup & Playing Time": {
        "metrics": ["Starts", "MP", "Min", "Min/MP", "Min%", "Min/90", "Mn/Start", "Subs", "Mn/Sub", "Compl MP", "unSub"],
        "description": "Analysis of player availability and expected playing time"
    },
    "2. Pressing & Defensive Intensity": {
        "metrics": ["TklAtt 3rd", "TklMid 3rd", "TklDef 3rd", "Tkl", "TklW", "Tkl chl", "Att chl", "Tkl% chl", "Lost chl", "Blocks", "TO Att", "TO Succ%", "Dispos", "Miscntrl"],
        "description": "Evaluation of pressing effectiveness and defensive work rate"
    },
    "3. Defensive Solidity": {
        "metrics": ["Tkl", "TklW", "Tkl% chl", "Tkl+Int", "Int", "Blocks", "Sh blk", "Clr", "Err sht", "Def Pen tch", "Def 3rd tch"],
        "description": "Assessment of defensive capabilities and reliability"
    },
    "4. Physical Aggression": {
        "metrics": ["Won", "Lost", "Won%", "Tkl chl", "Att chl", "Tkl% chl", "Fls", "CrdY", "CrdR", "2CrdY", "Err sht", "PKcon", "OG"],
        "description": "Physical presence and disciplinary considerations"
    },
    "5. Chance Creation (Attack)": {
        "metrics": ["Sh", "SoT", "SoT%", "Sh/90", "SoT/90", "xG", "npxG", "npxG/Sh", "G-xG", "KP", "xAG", "xA", "A-xAG", "P.F.1/3", "PPA", "CrsPA", "PrgP", "pn.PrgP", "PrgC"],
        "description": "Attacking threat and chance creation capabilities"
    },
    "6. Finishing & Goal Threat": {
        "metrics": ["Gls", "G-PK", "npxG", "np:G-xG", "PK", "PKatt", "Dist"],
        "description": "Goal-scoring ability and finishing quality"
    },
    "7. Build-Up Style": {
        "metrics": ["PrgP", "TotDist", "PrgDist", "SP.Cmp%", "MP.Cmp%", "LP.Cmp%", "PrgC", "PrgDist Car", "TotDist Car", "CPA", "F.1/3 Car", "SP.Att", "MP.Att", "LP.Att", "Dispos", "TO Succ%", "Miscntrl"],
        "description": "Ball progression and build-up play patterns"
    },
    "8. Creativity & Playmaking": {
        "metrics": ["SCA", "SCA90", "xAG", "xA", "KP", "SCA.P.Live", "SCA.P.Dead", "SCA_TO", "SCA_sh", "SCA_fld", "SCA_def", "PrgC", "Crs", "Sw", "CK", "InCK", "OutCK"],
        "description": "Creative output and playmaking abilities"
    },
    "9. Game Impact & Team Value": {
        "metrics": ["G+A", "G+A-PK", "pn.G+A", "pn.G+A-PK", "xG+xAG", "pn.npxG+xAG", "onG", "onGA", "GS-GA/90", "xGS-xGA", "xG On-Off"],
        "description": "Overall contribution and impact on team performance"
    },
    "10. Tactical Tendencies & Risk": {
        "metrics": ["Live tch", "TO Att", "TO Succ%", "TO Tkld%", "Mid 3rd tch", "Att 3rd tch", "Att Pen tch", "PrgC", "PrgP", "LP.Att", "LP.Cmp%", "TotDist", "Dead.P", "FK", "CK", "PK"],
        "description": "Tactical behavior and risk-taking patterns"
    },
    "11. Team Cohesion & Consistency": {
        "metrics": ["Starts", "Compl MP", "Min%", "Err sht", "PKcon", "OG"],
        "description": "Reliability and consistency metrics"
    },
    "12. Opposition-Specific Matchups": {
        "metrics": ["TO Succ%", "Dispos", "Miscntrl", "Crs", "Blocks", "Won%"],
        "description": "Head-to-head matchup considerations"
    },
    "13. Trend & Form": {
        "metrics": ["Min", "90s"],
        "description": "Recent performance trends and form analysis (based on full season data)"
    },
    "14. Discipline & Mentality": {
        "metrics": ["CrdY", "CrdR", "2CrdY", "Fls", "Fld", "PKcon"],
        "description": "Disciplinary record and mental approach"
    }
}

# Chart type mapping
METRIC_TO_CHART_TYPE = {
    'Gls': 'bar', 'Ast': 'bar', 'Sh': 'bar', 'SoT': 'bar', 'PrgP': 'bar',
    'Tkl': 'bar', 'Int': 'bar', 'Blocks': 'bar', 'Won%': 'pie', 'Min%': 'horizontal_bar',
    'SoT%': 'pie', 'Tkl% chl': 'pie', 'TO Succ%': 'pie', 'SP.Cmp%': 'pie',
    'MP.Cmp%': 'pie', 'LP.Cmp%': 'pie', 'CrdY': 'bar', 'CrdR': 'bar',
    'xG': 'bar', 'xAG': 'bar', 'SCA': 'bar', 'PrgC': 'bar', 'TotDist': 'bar'
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filename):
    """Loads and preprocesses the Premier League data"""
    try:
        df = pd.read_csv(filename)
        # Fix potential column name mismatch
        if "Blocks" not in df.columns and "Blocks.1" in df.columns:
             df = df.rename(columns={"Blocks.1": "Blocks"})
        
        # Convert all potential metric columns to numeric, coercing errors
        # This is heavy but ensures that ML models and charts don't fail
        for category in METRIC_CATEGORIES.values():
            for metric in category['metrics']:
                if metric in df.columns:
                    df[metric] = pd.to_numeric(df[metric], errors='coerce')
        
        print(f"Successfully loaded '{filename}' with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except FileNotFoundError:
        print(f"Error: Data file '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ============================================================================
# MACHINE LEARNING MODELS
# ============================================================================

class PerformancePredictor:
    """ML model to predict top performers for specific metrics"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        
    def _get_feature_columns(self, metric):
        """Define relevant features for each metric category"""
        
        # Feature sets for different metric types
        attacking_features = ['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'xAG', 'KP', 'PrgP', 'PrgC', 'Min', '90s']
        defensive_features = ['Tkl', 'Int', 'Blocks', 'TklW', 'Tkl+Int', 'Clr', 'Def 3rd tch', 'Min', '90s']
        passing_features = ['TP.Cmp', 'Att', 'Cmp%', 'TotDist', 'PrgDist', 'KP', 'PrgP', 'Min', '90s']
        physical_features = ['Won', 'Lost', 'Won%', 'Tkl chl', 'Att chl', 'Fls', 'Min', '90s']
        playmaking_features = ['SCA', 'xAG', 'KP', 'PrgC', 'PrgP', 'Ast', 'xA', 'Min', '90s']
        
        # Map metrics to feature sets
        feature_map = {
            'Gls': attacking_features,
            'Ast': playmaking_features,
            'xG': attacking_features,
            'Tkl': defensive_features,
            'Int': defensive_features,
            'PrgP': passing_features,
            'SCA': playmaking_features,
            'Won%': physical_features,
            'Min%': ['Starts', 'MP', 'Min', 'Compl MP', 'Subs'],
        }
        
        # Return specific features or default attacking features
        default_features = ['Min', '90s', 'MP', 'Starts']
        selected_features = feature_map.get(metric, default_features)
        
        # Always add basic playing time stats
        for f in default_features:
            if f not in selected_features:
                selected_features.append(f)
                
        return selected_features
    
    def _prepare_features(self, df, metric, feature_cols):
        """Prepare feature matrix for training"""
        available_cols = [col for col in feature_cols if col in df.columns and col != metric]
        
        if not available_cols:
            available_cols = ['Min', '90s', 'MP']
            available_cols = [col for col in available_cols if col in df.columns]
        
        X = df[available_cols].fillna(0)
        
        # Handle percentage columns that might be strings
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        return X, available_cols
    
    def train_model(self, df, metric):
        """Train a model to predict performance for a specific metric"""
        
        if metric not in df.columns:
            return None
        
        # Get target variable
        y = pd.to_numeric(df[metric], errors='coerce').fillna(0)
        
        if y.sum() == 0 or len(y) < 10:
            return None
        
        # Get features
        feature_cols = self._get_feature_columns(metric)
        X, used_cols = self._prepare_features(df, metric, feature_cols)
        
        if X.empty or len(used_cols) == 0:
            return None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_scaled, y)
        
        # Store model and scaler
        self.models[metric] = model
        self.scalers[metric] = scaler
        self.feature_importances[metric] = dict(zip(X.columns, model.feature_importances_))
        
        return model
    
    def predict_top_performers(self, all_data, team_data, metric, n=3):
        """
        Predict top N performers for a specific metric.
        Trains on all_data, predicts on team_data.
        """
        
        if metric not in self.models:
            # Train model if not exists, using all available data
            print(f"Training model for {metric}...")
            self.train_model(all_data, metric)
        
        if metric not in self.models:
            # Fallback to actual values if model training failed
            valid_data = pd.to_numeric(team_data[metric], errors='coerce').fillna(0)
            top_indices = valid_data.nlargest(n).index
            return team_data.loc[top_indices, ['Player', 'Pos', metric]].to_dict('records')
        
        model = self.models[metric]
        scaler = self.scalers[metric]
        
        # Prepare features from team_data
        feature_cols = self._get_feature_columns(metric)
        X_team, _ = self._prepare_features(team_data, metric, feature_cols)
        
        if X_team.empty:
             # Fallback
            valid_data = pd.to_numeric(team_data[metric], errors='coerce').fillna(0)
            top_indices = valid_data.nlargest(n).index
            return team_data.loc[top_indices, ['Player', 'Pos', metric]].to_dict('records')

        X_team_scaled = scaler.transform(X_team)
        
        # Predict
        predictions = model.predict(X_team_scaled)
        df_copy = team_data.copy()
        df_copy[f'{metric}_predicted'] = predictions
        
        # Get top N based on prediction
        top_indices = df_copy.nlargest(n, f'{metric}_predicted').index
        
        result = []
        for idx in top_indices:
            player_data = {
                'Player': team_data.loc[idx, 'Player'],
                'Pos': team_data.loc[idx, 'Pos'],
                f'{metric}': team_data.loc[idx, metric],
                f'{metric}_predicted': df_copy.loc[idx, f'{metric}_predicted']
            }
            result.append(player_data)
        
        return result


class ImpactAnalyzer:
    """ML model to analyze player impact on match outcomes"""
    
    def __init__(self):
        self.impact_models = {}
        self.baseline_metrics = {}
        
    def analyze_player_impact(self, df, player_name, metric_category):
        """Analyze how a player's performance impacts team success"""
        
        # Find player
        player_data = df[df['Player'] == player_name]
        
        if player_data.empty:
            return {
                'player': player_name,
                'position': 'N/A',
                'impact_score': 0,
                'category': metric_category,
                'success_probability': 0,
                'key_strengths': [],
                'risk_factors': []
            }
        
        player = player_data.iloc[0]
        
        # Calculate impact scores based on category
        impact_scores = {}
        
        # Define category-specific metrics
        if "Playing Time" in metric_category:
            impact_scores = self._analyze_playing_time_impact(player)
        elif "Pressing" in metric_category or "Defensive Intensity" in metric_category:
            impact_scores = self._analyze_defensive_impact(player)
        elif "Defensive Solidity" in metric_category:
            impact_scores = self._analyze_defensive_solidity_impact(player)
        elif "Physical Aggression" in metric_category:
            impact_scores = self._analyze_physical_impact(player)
        elif "Chance Creation" in metric_category:
            impact_scores = self._analyze_attacking_impact(player)
        elif "Finishing" in metric_category or "Goal Threat" in metric_category:
            impact_scores = self._analyze_finishing_impact(player)
        elif "Build-Up" in metric_category:
            impact_scores = self._analyze_buildup_impact(player)
        elif "Creativity" in metric_category or "Playmaking" in metric_category:
            impact_scores = self._analyze_creativity_impact(player)
        elif "Game Impact" in metric_category:
            impact_scores = self._analyze_overall_impact(player)
        elif "Discipline" in metric_category:
            impact_scores = self._analyze_discipline_impact(player)
        else:
            impact_scores = self._analyze_general_impact(player)
        
        # Calculate success probability
        success_prob = self._calculate_success_probability(impact_scores)
        
        # Identify strengths and risks
        strengths = [k for k, v in impact_scores.items() if v > 0.6][:3]
        risks = [k for k, v in impact_scores.items() if v < 0.3][:3]
        
        return {
            'player': player_name,
            'position': player.get('Pos', 'N/A'),
            'impact_score': np.mean(list(impact_scores.values())) if impact_scores else 0,
            'category': metric_category,
            'success_probability': success_prob,
            'detailed_impacts': impact_scores,
            'key_strengths': strengths,
            'risk_factors': risks
        }
    
    def _analyze_playing_time_impact(self, player):
        """Analyze playing time metrics impact"""
        min_pct = self._normalize_value(player.get('Min%', 0), 0, 100)
        starts = self._normalize_value(player.get('Starts', 0), 0, 38)
        complete = self._normalize_value(player.get('Compl MP', 0), 0, 30)
        
        return {
            'availability': min_pct,
            'starting_role': starts,
            'endurance': complete,
            'consistency': (starts + complete) / 2
        }
    
    def _analyze_defensive_impact(self, player):
        """Analyze defensive pressing impact"""
        tackles = self._normalize_value(player.get('Tkl', 0), 0, 80)
        tackle_success = self._normalize_value(player.get('Tkl% chl', 0), 0, 100)
        interceptions = self._normalize_value(player.get('Int', 0), 0, 50)
        blocks = self._normalize_value(player.get('Blocks', 0), 0, 30)
        
        return {
            'tackle_volume': tackles,
            'tackle_efficiency': tackle_success,
            'interception_rate': interceptions,
            'blocking_ability': blocks,
            'overall_defense': (tackles + tackle_success + interceptions) / 3
        }
    
    def _analyze_defensive_solidity_impact(self, player):
        """Analyze defensive solidity"""
        tkl_int = self._normalize_value(player.get('Tkl+Int', 0), 0, 120)
        clearances = self._normalize_value(player.get('Clr', 0), 0, 200)
        def_actions = self._normalize_value(player.get('Def 3rd tch', 0), 0, 800)
        errors = 1 - self._normalize_value(player.get('Err sht', 0), 0, 5)
        
        return {
            'defensive_actions': tkl_int,
            'clearance_rate': clearances,
            'defensive_presence': def_actions,
            'reliability': errors,
            'solidity_score': (tkl_int + clearances + errors) / 3
        }
    
    def _analyze_physical_impact(self, player):
        """Analyze physical presence"""
        duels_won = self._normalize_value(player.get('Won%', 0), 0, 100)
        aerial_won = self._normalize_value(player.get('Won', 0), 0, 100) # 'Won' is aerials won
        aggression = self._normalize_value(player.get('Fls', 0), 0, 60)
        discipline = 1 - self._normalize_value(player.get('CrdY', 0), 0, 12)
        
        return {
            'duel_success': duels_won,
            'aerial_dominance': aerial_won,
            'aggression_level': aggression,
            'discipline': discipline,
            'physical_score': (duels_won + aerial_won + discipline) / 3
        }
    
    def _analyze_attacking_impact(self, player):
        """Analyze attacking and chance creation"""
        shots = self._normalize_value(player.get('Sh', 0), 0, 120)
        xg = self._normalize_value(player.get('xG', 0), 0, 25)
        key_passes = self._normalize_value(player.get('KP', 0), 0, 70)
        prog_passes = self._normalize_value(player.get('PrgP', 0), 0, 200)
        
        return {
            'shot_volume': shots,
            'scoring_threat': xg,
            'key_pass_ability': key_passes,
            'progression': prog_passes,
            'attacking_score': (shots + xg + key_passes) / 3
        }
    
    def _analyze_finishing_impact(self, player):
        """Analyze finishing quality"""
        goals = self._normalize_value(player.get('Gls', 0), 0, 30)
        npxg = self._normalize_value(player.get('npxG', 0), 0, 20)
        g_minus_xg = player.get('G-xG', 0)
        finishing = self._normalize_value(g_minus_xg + 10, 0, 20)  # Offset to make positive
        
        return {
            'goal_output': goals,
            'expected_goals': npxg,
            'finishing_quality': finishing,
            'clinical_rating': (goals + finishing) / 2
        }
    
    def _analyze_buildup_impact(self, player):
        """Analyze build-up play"""
        pass_comp = self._normalize_value(player.get('Cmp%', 0), 0, 100)
        prog_dist = self._normalize_value(player.get('PrgDist', 0), 0, 5000)
        prog_carries = self._normalize_value(player.get('PrgC', 0), 0, 100)
        total_dist = self._normalize_value(player.get('TotDist', 0), 0, 20000)
        
        return {
            'passing_accuracy': pass_comp,
            'progressive_distance': prog_dist,
            'carrying_ability': prog_carries,
            'passing_volume': total_dist,
            'buildup_score': (pass_comp + prog_dist + prog_carries) / 3
        }
    
    def _analyze_creativity_impact(self, player):
        """Analyze creativity and playmaking"""
        sca = self._normalize_value(player.get('SCA', 0), 0, 120)
        xag = self._normalize_value(player.get('xAG', 0), 0, 15)
        key_passes = self._normalize_value(player.get('KP', 0), 0, 70)
        assists = self._normalize_value(player.get('Ast', 0), 0, 15)
        
        return {
            'shot_creating_actions': sca,
            'expected_assists': xag,
            'key_passes': key_passes,
            'actual_assists': assists,
            'creativity_score': (sca + xag + key_passes) / 3
        }
    
    def _analyze_overall_impact(self, player):
        """Analyze overall game impact"""
        g_plus_a = self._normalize_value(player.get('G+A', 0), 0, 40)
        xg_plus_xag = self._normalize_value(player.get('xG+xAG', 0), 0, 35)
        xg_diff = player.get('xG On-Off', 0)
        team_impact = self._normalize_value(xg_diff + 10, 0, 20)
        
        return {
            'goal_contributions': g_plus_a,
            'expected_contributions': xg_plus_xag,
            'team_impact': team_impact,
            'overall_value': (g_plus_a + xg_plus_xag + team_impact) / 3
        }
    
    def _analyze_discipline_impact(self, player):
        """Analyze discipline"""
        yellow_cards = 1 - self._normalize_value(player.get('CrdY', 0), 0, 12)
        red_cards = 1 - self._normalize_value(player.get('CrdR', 0), 0, 2)
        fouls = 1 - self._normalize_value(player.get('Fls', 0), 0, 60)
        
        return {
            'yellow_card_discipline': yellow_cards,
            'red_card_risk': red_cards,
            'foul_discipline': fouls,
            'overall_discipline': (yellow_cards + red_cards + fouls) / 3
        }
    
    def _analyze_general_impact(self, player):
        """General impact analysis"""
        mins = self._normalize_value(player.get('Min', 0), 0, 3400)
        return {
            'playing_time': mins,
            'general_contribution': mins
        }
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize value to 0-1 scale"""
        try:
            value = float(value)
            if max_val == min_val:
                return 0.5
            normalized = (value - min_val) / (max_val - min_val)
            return max(0, min(1, normalized))
        except (ValueError, TypeError):
            return 0.5
    
    def _calculate_success_probability(self, impact_scores):
        """Calculate overall success probability based on impact scores"""
        if not impact_scores:
            return 50.0
        
        avg_score = np.mean(list(impact_scores.values()))
        # Convert to percentage with some noise reduction
        probability = (avg_score * 70) + 15  # Range: 15-85%
        return round(probability, 1)

# ============================================================================
# CHART GENERATION
# ============================================================================

def create_chart(data, metric, chart_type, output_path):
    """Create and save a chart for a specific metric"""
    
    plt.figure(figsize=(10, 6))
    
    try:
        if data.empty or metric not in data.columns:
            # Create empty chart with message
            plt.text(0.5, 0.5, f'No data available for {metric}', 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
        else:
            # Clean data
            plot_data = data[['Player', metric]].copy()
            plot_data[metric] = pd.to_numeric(plot_data[metric], errors='coerce')
            plot_data = plot_data.dropna(subset=[metric])
            plot_data = plot_data[plot_data[metric] > 0] # Show only players with the stat
            plot_data = plot_data.nlargest(10, metric).sort_values(metric, ascending=False)
            
            if plot_data.empty:
                plt.text(0.5, 0.5, f'No valid data for {metric}', 
                        ha='center', va='center', fontsize=14)
                plt.axis('off')
            else:
                if chart_type == 'bar':
                    sns.barplot(x='Player', y=metric, data=plot_data, palette="Blues_d")
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel(metric)
                    plt.xlabel("Player")
                    plt.title(f'Top Players - {metric}')
                    
                elif chart_type == 'horizontal_bar':
                    sns.barplot(x=metric, y='Player', data=plot_data, palette="Greens_d")
                    plt.xlabel(metric)
                    plt.ylabel("Player")
                    plt.title(f'Top Players - {metric}')
                    
                elif chart_type == 'pie':
                    if len(plot_data) > 5:
                        plot_data = plot_data.head(5)
                    plt.pie(plot_data[metric], labels=plot_data['Player'], autopct='%1.1f%%',
                            colors=sns.color_palette("pastel"))
                    plt.title(f'{metric} Distribution (Top 5)')
                    
                else:  # default to bar
                    sns.barplot(x='Player', y=metric, data=plot_data, palette="rocket")
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel(metric)
                    plt.xlabel("Player")
                    plt.title(f'Top Players - {metric}')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating chart for {metric}: {e}")
        # Create error chart
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Error generating chart for {metric}', 
                ha='center', va='center', fontsize=12, color='red')
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return False

# ============================================================================
# PDF GENERATION
# ============================================================================

def create_report(home_team, away_team, data, predictor, analyzer):
    """Generate the complete scouting report PDF"""
    
    filename = f"Scout_Report_{away_team.replace(' ', '_')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4,
                            leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    story = []
    chart_paths = [] # To store chart paths for cleanup
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        spaceBefore=12,
        borderPadding=5,
        borderWidth=1,
        borderColor=colors.HexColor('#e0e0e0'),
        backColor=colors.HexColor('#f7f9fc')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#2e5090'),
        spaceAfter=6,
        spaceBefore=10,
        borderBottomWidth=1,
        borderBottomColor=colors.HexColor('#cccccc'),
        paddingBottom=2
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    insight_style = ParagraphStyle(
        'Insight',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceAfter=6,
        borderPadding=8,
        backColor=colors.HexColor('#fafafa'),
        borderWidth=1,
        borderColor=colors.HexColor('#f0f0f0')
    )

    
    # Cover Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph(f"SCOUTING REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"<b>{away_team}</b>", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Prepared for: <b>{home_team}</b>", 
                          ParagraphStyle('Center', parent=styles['Normal'], alignment=TA_CENTER, fontSize=14)))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", 
                          ParagraphStyle('Center', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12)))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Powered by Advanced ML Analytics", 
                          ParagraphStyle('Center', parent=styles['Normal'], alignment=TA_CENTER, 
                                       fontSize=10, textColor=colors.grey)))
    
    story.append(PageBreak())
    
    # Filter data for away team
    team_data = data[data['Squad'] == away_team].copy().reset_index(drop=True)
    
    if team_data.empty:
        story.append(Paragraph(f"No data available for {away_team}", body_style))
        doc.build(story)
        return filename
    
    # Process each category
    for category_name, category_info in METRIC_CATEGORIES.items():
        story.append(Paragraph(category_name, heading_style))
        story.append(Paragraph(category_info['description'], body_style))
        story.append(Spacer(1, 0.2*inch))
        
        metrics = category_info['metrics']
        
        for metric in metrics:
            if metric not in team_data.columns:
                continue
            
            # Check if metric has any valid data
            metric_data = pd.to_numeric(team_data[metric], errors='coerce')
            if metric_data.isna().all() or metric_data.sum() == 0:
                continue
            
            # --- This is the core logic that was missing ---
            
            # 1. Add sub-heading
            story.append(Paragraph(f"Metric Analysis: {metric}", subheading_style))
            
            # 2. Get Top 3 Performers (ML Model 1)
            top_performers = predictor.predict_top_performers(data, team_data, metric, n=3)
            
            story.append(Paragraph(f"<b>Top Predicted Performers ({metric}):</b>", body_style))
            performers_text = []
            if not top_performers:
                performers_text.append("No performers found.")
            else:
                for p in top_performers:
                    pred_val = p.get(f'{metric}_predicted', p.get(metric, 0))
                    actual_val = p.get(metric, 0)
                    performers_text.append(f" - <b>{p['Player']} ({p['Pos']})</b> | Actual: {actual_val:.1f} | Predicted: {pred_val:.1f}")
            story.append(Paragraph("<br/>".join(performers_text), body_style))
            story.append(Spacer(1, 0.1*inch))

            # 3. Get Key Insight (ML Model 2)
            story.append(Paragraph(f"<b>Key Impact Analysis:</b>", body_style))
            if top_performers:
                top_player_name = top_performers[0]['Player']
                impact = analyzer.analyze_player_impact(team_data, top_player_name, category_name)
                
                insight_text = (f"Top performer <b>{top_player_name} ({impact['position']})</b> shows a "
                                f"<b>{impact['success_probability']:.1f}%</b> success probability in this category. <br/>"
                                f"<b>Key Strengths:</b> {', '.join(impact['key_strengths']) or 'N/A'} <br/>"
                                f"<b>Risk Factors:</b> {', '.join(impact['risk_factors']) or 'N/A'}")
            else:
                insight_text = "No top performer to analyze for impact."
            story.append(Paragraph(insight_text, insight_style))
            story.append(Spacer(1, 0.1*inch))

            # 4. Generate Chart
            chart_path = f"chart_{away_team.replace(' ', '_')}_{metric.replace('%', 'pct').replace('/', '_')}.png"
            chart_paths.append(chart_path) # Add to cleanup list
            chart_type = METRIC_TO_CHART_TYPE.get(metric, 'bar')
            create_chart(team_data, metric, chart_type, chart_path)

            # 5. Embed Chart
            try:
                img = Image(chart_path, width=5.5*inch, height=3.6*inch)
                img.hAlign = 'CENTER'
                story.append(img)
            except Exception as e:
                story.append(Paragraph(f"[Could not load chart for {metric}: {e}]", body_style))
            
            story.append(Spacer(1, 0.2*inch))
            # --- End of core logic ---

        story.append(PageBreak())
    
    # Build the PDF
    try:
        doc.build(story)
        print(f"\nSuccessfully generated report: {filename}")
    except Exception as e:
        print(f"Error building PDF: {e}")

    # Cleanup generated chart files
    print("Cleaning up chart files...")
    for path in chart_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Could not remove chart file {path}: {e}")
                
    return filename

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting Scouting Report Generator...")
    
    # --- Configuration ---
    # The home team is just for the cover page.
    HOME_TEAM = "Arsenal"
    AWAY_TEAM = "Chelsea"
    DATA_FILE = "C:\\Users\\Prasiddha\\OneDrive\\Desktop\\Capstone\\fbref\\Premier_League.csv"
    # --- End Configuration ---
    
    main_df = load_data(DATA_FILE)
    
    if main_df is not None:
        # Check if away team is in the data
        if AWAY_TEAM not in main_df['Squad'].unique():
            print(f"Error: Team '{AWAY_TEAM}' not found in the 'Squad' column of {DATA_FILE}.")
            print(f"Available teams: {main_df['Squad'].unique()}")
        else:
            # Initialize ML models
            predictor = PerformancePredictor()
            analyzer = ImpactAnalyzer()
            
            print(f"Instances created. Generating report for {AWAY_TEAM}...")
            
            # Run the report generation
            create_report(HOME_TEAM, AWAY_TEAM, main_df, predictor, analyzer)
            
            print(f"Report generation for {AWAY_TEAM} complete.")
    else:
        print(f"Failed to load data from {DATA_FILE}. Exiting.")
        
    print("Script finished.")