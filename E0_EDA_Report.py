import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings

# Suppress warnings for a cleaner report generation process
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (11, 8.5) # Standard A4-like size
plt.rcParams['font.size'] = 10

def create_title_page(pdf, title, subtitle=""):
    """Create a title page for the PDF report"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.6, title, ha='center', va='center', fontsize=28, fontweight='bold')
    if subtitle:
        ax.text(0.5, 0.5, subtitle, ha='center', va='center', fontsize=16)
    ax.text(0.5, 0.4, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
            ha='center', va='center', fontsize=12, style='italic')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def add_table_to_pdf(pdf, df, title):
    """Add a DataFrame as a table to the PDF"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Create table
    # Handle both indexed and non-indexed DataFrames for flexibility
    if isinstance(df.index, pd.RangeIndex):
        table_data = df.values
        col_labels = df.columns.tolist()
    else:
        table_data = df.reset_index().values
        col_labels = df.reset_index().columns.tolist()

    table = ax.table(cellText=table_data, colLabels=col_labels, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8) # Adjust scale for better row height
    
    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        
    # Style index column if it's not a RangeIndex
    if not isinstance(df.index, pd.RangeIndex):
         for i in range(1, len(table_data) + 1):
            table[(i, 0)].set_facecolor('#F0F0F0')
            table[(i, 0)].set_text_props(weight='bold')
            
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def add_text_page(pdf, title, text_content):
    """Add a text page to the PDF"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(0.1, 0.85, text_content, ha='left', va='top', fontsize=10, 
            wrap=True, family='monospace')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================
print("Starting EDA Report Generation...")

# Load data
try:
    df = pd.read_csv('C:\\Users\\Prasiddha\\OneDrive\\Desktop\\Capstone\\Feature selection\\E0.csv')
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
except FileNotFoundError:
    print("Error: E0.csv not found.")
    print("Please make sure the file is in the same directory as the script.")
    exit()

# Parse dates and create date features
# Using errors='coerce' will turn unparseable dates into NaT
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
df['Month'] = df['Date'].dt.month
df['MonthName'] = df['Date'].dt.strftime('%B')

# Create PDF
with PdfPages('E0_EDA_Report.pdf') as pdf:
    
    # ============================================================================
    # TITLE PAGE
    # ============================================================================
    create_title_page(pdf, "English Premier League", 
                      "Comprehensive Exploratory Data Analysis Report\n2024-2025 Season")
    
    # ============================================================================
    # SECTION 1: DATA OVERVIEW & PREPROCESSING
    # ============================================================================
    print("Generating Section 1: Data Overview...")
    
    # 1.1 First 10 rows
    add_table_to_pdf(pdf, df.head(10), "1.1 Data Preview - First 10 Rows")
    
    # 1.2 Data Info
    # Capture .info() output
    from io import StringIO
    buffer = StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    add_text_page(pdf, "1.2 Dataset Information (df.info())", info_text)
    
    # 1.3 Missing Data Summary
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_data) > 0:
        add_table_to_pdf(pdf, missing_data.head(30), "1.3 Missing Data Summary (Top 30)")
    else:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.5, "No Missing Data Found!", ha='center', va='center', 
                fontsize=20, fontweight='bold', color='green')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # 1.4 Missing Data Heatmap
    fig, ax = plt.subplots(figsize=(11, 8.5))
    # Select columns with missing data for better visualization
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    if len(cols_with_missing) > 0:
        sns.heatmap(df[cols_with_missing].isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
        ax.set_title("1.4 Missing Data Pattern Heatmap", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Columns", fontsize=12)
        ax.set_ylabel("Rows", fontsize=12)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, "No Missing Data to Visualize", ha='center', va='center', fontsize=16)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 1.5 Summary Statistics - Numerical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary_stats = df[numeric_cols].describe().T
        summary_stats = summary_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        summary_stats = summary_stats.round(2)
        
        # Split into multiple pages if needed
        chunk_size = 25
        for i in range(0, len(summary_stats), chunk_size):
            chunk = summary_stats.iloc[i:i+chunk_size]
            add_table_to_pdf(pdf, chunk, f"1.5 Summary Statistics - Numerical (Part {i//chunk_size + 1})")
    
    # 1.6 Summary Statistics - Categorical
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(cat_cols) > 0:
        cat_summary = df[cat_cols].describe().T
        cat_summary = cat_summary[['count', 'unique', 'top', 'freq']]
        add_table_to_pdf(pdf, cat_summary, "1.6 Summary Statistics - Categorical")
    
    # ============================================================================
    # SECTION 2: MATCH RESULTS & GOALS ANALYSIS
    # ============================================================================
    print("Generating Section 2: Match Results...")
    
    # 2.1 Full-Time Results Distribution
    ftr_counts = df['FTR'].value_counts()
    ftr_pct = (ftr_counts / len(df) * 100).round(2)
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    bars = ax.bar(['Home Win', 'Draw', 'Away Win'], 
                  [ftr_counts.get('H', 0), ftr_counts.get('D', 0), ftr_counts.get('A', 0)],
                  color=['#4472C4', '#FFC000', '#FF5733'])
    ax.set_title("2.1 Full-Time Results Distribution", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Match Result", fontsize=12)
    ax.set_ylabel("Number of Matches", fontsize=12)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        result_key = ['H', 'D', 'A'][i]
        pct = ftr_pct.get(result_key, 0)
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({pct}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 2.2 Half-Time Results Distribution
    htr_counts = df['HTR'].value_counts()
    htr_pct = (htr_counts / len(df) * 100).round(2)
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    bars = ax.bar(['Home Win', 'Draw', 'Away Win'], 
                  [htr_counts.get('H', 0), htr_counts.get('D', 0), htr_counts.get('A', 0)],
                  color=['#70AD47', '#FFC000', '#C55A11'])
    ax.set_title("2.2 Half-Time Results Distribution", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Half-Time Result", fontsize=12)
    ax.set_ylabel("Number of Matches", fontsize=12)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        result_key = ['H', 'D', 'A'][i]
        pct = htr_pct.get(result_key, 0)
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({pct}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 2.3 HT vs FT Results Heatmap
    ht_ft_crosstab = pd.crosstab(df['HTR'], df['FTR'])
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    sns.heatmap(ht_ft_crosstab, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title("2.3 Half-Time vs Full-Time Results Cross-Tabulation", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Full-Time Result (H=Home Win, D=Draw, A=Away Win)", fontsize=12)
    ax.set_ylabel("Half-Time Result (H=Home Lead, D=Draw, A=Away Lead)", fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 2.4 Goal Distributions
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['GoalDifference'] = df['FTHG'] - df['FTAG']
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("2.4 Goal Distributions", fontsize=16, fontweight='bold')
    
    # Home Goals
    axes[0, 0].hist(df['FTHG'], bins=range(0, df['FTHG'].max()+2), 
                    color='#4472C4', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title("Full-Time Home Goals", fontweight='bold')
    axes[0, 0].set_xlabel("Goals Scored", fontsize=10)
    axes[0, 0].set_ylabel("Frequency", fontsize=10)
    axes[0, 0].axvline(df['FTHG'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["FTHG"].mean():.2f}')
    axes[0, 0].legend()
    
    # Away Goals
    axes[0, 1].hist(df['FTAG'], bins=range(0, df['FTAG'].max()+2), 
                    color='#FF5733', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("Full-Time Away Goals", fontweight='bold')
    axes[0, 1].set_xlabel("Goals Scored", fontsize=10)
    axes[0, 1].set_ylabel("Frequency", fontsize=10)
    axes[0, 1].axvline(df['FTAG'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["FTAG"].mean():.2f}')
    axes[0, 1].legend()
    
    # Total Goals
    axes[1, 0].hist(df['TotalGoals'], bins=range(0, df['TotalGoals'].max()+2), 
                    color='#70AD47', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Total Goals per Match", fontweight='bold')
    axes[1, 0].set_xlabel("Total Goals", fontsize=10)
    axes[1, 0].set_ylabel("Frequency", fontsize=10)
    axes[1, 0].axvline(df['TotalGoals'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["TotalGoals"].mean():.2f}')
    axes[1, 0].legend()
    
    # Goal Difference
    axes[1, 1].hist(df['GoalDifference'], bins=range(df['GoalDifference'].min(), 
                     df['GoalDifference'].max()+2), color='#FFC000', 
                     edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Goal Difference (Home - Away)", fontweight='bold')
    axes[1, 1].set_xlabel("Goal Difference", fontsize=10)
    axes[1, 1].set_ylabel("Frequency", fontsize=10)
    axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=2)
    axes[1, 1].axvline(df['GoalDifference'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["GoalDifference"].mean():.2f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 2.5 Goals Box Plot
    goals_data = pd.DataFrame({
        'Goals': list(df['FTHG']) + list(df['FTAG']),
        'Team': ['Home']*len(df) + ['Away']*len(df)
    })
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    sns.boxplot(data=goals_data, x='Team', y='Goals', palette=['#4472C4', '#FF5733'], ax=ax)
    ax.set_title("2.5 Home vs Away Goals Distribution", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Team Type", fontsize=12)
    ax.set_ylabel("Goals Scored", fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # SECTION 3: TEAM-LEVEL ANALYSIS
    # ============================================================================
    print("Generating Section 3: Team Analysis...")
    
    # Create team-level features
    df['HomeWin'] = (df['FTR'] == 'H').astype(int)
    df['AwayWin'] = (df['FTR'] == 'A').astype(int)
    df['DrawResult'] = (df['FTR'] == 'D').astype(int)
    
    # Home team stats
    home_stats = df.groupby('HomeTeam').agg({
        'FTHG': ['sum', 'mean'],
        'FTAG': ['sum', 'mean'],
        'HomeWin': 'sum',
        'DrawResult': 'sum'
    }).reset_index()
    home_stats.columns = ['Team', 'HomeGoalsScored', 'AvgHomeGoalsScored', 
                          'HomeGoalsConceded', 'AvgHomeGoalsConceded', 'HomeWins', 'HomeDraws']
    
    # Away team stats
    away_stats = df.groupby('AwayTeam').agg({
        'FTAG': ['sum', 'mean'],
        'FTHG': ['sum', 'mean'],
        'AwayWin': 'sum',
        'DrawResult': 'sum'
    }).reset_index()
    away_stats.columns = ['Team', 'AwayGoalsScored', 'AvgAwayGoalsScored', 
                          'AwayGoalsConceded', 'AvgAwayGoalsConceded', 'AwayWins', 'AwayDraws']
    
    # Merge stats
    team_stats = pd.merge(home_stats, away_stats, on='Team')
    team_stats['TotalWins'] = team_stats['HomeWins'] + team_stats['AwayWins']
    team_stats['TotalGoalsScored'] = team_stats['HomeGoalsScored'] + team_stats['AwayGoalsScored']
    team_stats['TotalGoalsConceded'] = team_stats['HomeGoalsConceded'] + team_stats['AwayGoalsConceded']
    team_stats['GoalDifference'] = team_stats['TotalGoalsScored'] - team_stats['TotalGoalsConceded']
    
    # 3.1 Top 10 Teams by Wins
    top_wins = team_stats[['Team', 'TotalWins', 'HomeWins', 'AwayWins']].sort_values(
        'TotalWins', ascending=False).head(10).reset_index(drop=True)
    add_table_to_pdf(pdf, top_wins, "3.1 Top 10 Teams by Total Wins")
    
    # 3.2 Top 10 Teams by Goals Scored
    top_goals = team_stats[['Team', 'TotalGoalsScored', 'HomeGoalsScored', 
                            'AwayGoalsScored']].sort_values('TotalGoalsScored', 
                            ascending=False).head(10).reset_index(drop=True)
    add_table_to_pdf(pdf, top_goals, "3.2 Top 10 Teams by Total Goals Scored")
    
    # 3.3 Top 10 Teams by Goals Conceded
    top_conceded = team_stats[['Team', 'TotalGoalsConceded', 'HomeGoalsConceded', 
                               'AwayGoalsConceded']].sort_values('TotalGoalsConceded', 
                               ascending=True).head(10).reset_index(drop=True)
    add_table_to_pdf(pdf, top_conceded, "3.3 Top 10 Teams by Fewest Goals Conceded")
    
    # 3.4 Home vs Away Goals Scored
    fig, ax = plt.subplots(figsize=(11, 8.5))
    x = np.arange(len(team_stats))
    width = 0.35
    
    teams_sorted = team_stats.sort_values('TotalGoalsScored', ascending=False)
    ax.bar(x - width/2, teams_sorted['AvgHomeGoalsScored'], width, 
           label='Home', color='#4472C4', alpha=0.8)
    ax.bar(x + width/2, teams_sorted['AvgAwayGoalsScored'], width, 
           label='Away', color='#FF5733', alpha=0.8)
    
    ax.set_title("3.4 Average Goals Scored: Home vs Away (by Team)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Teams (sorted by total goals)", fontsize=12)
    ax.set_ylabel("Average Goals per Match", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(teams_sorted['Team'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 3.5 Home vs Away Goals Conceded
    fig, ax = plt.subplots(figsize=(11, 8.5))
    teams_sorted = team_stats.sort_values('TotalGoalsConceded', ascending=True)
    ax.bar(x - width/2, teams_sorted['AvgHomeGoalsConceded'], width, 
           label='Home', color='#70AD47', alpha=0.8)
    ax.bar(x + width/2, teams_sorted['AvgAwayGoalsConceded'], width, 
           label='Away', color='#C55A11', alpha=0.8)
    
    ax.set_title("3.5 Average Goals Conceded: Home vs Away (by Team)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Teams (sorted by fewest goals conceded)", fontsize=12)
    ax.set_ylabel("Average Goals Conceded per Match", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(teams_sorted['Team'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 3.6 Goal Difference by Team
    fig, ax = plt.subplots(figsize=(11, 8.5))
    teams_sorted = team_stats.sort_values('GoalDifference', ascending=False)
    colors = ['green' if x > 0 else 'red' for x in teams_sorted['GoalDifference']]
    ax.barh(teams_sorted['Team'], teams_sorted['GoalDifference'], color=colors, alpha=0.7)
    ax.set_title("3.6 Goal Difference by Team (Goals Scored - Goals Conceded)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Goal Difference", fontsize=12)
    ax.set_ylabel("Team", fontsize=12)
    ax.axvline(0, color='black', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # SECTION 4: MATCH STATISTICS DEEP DIVE
    # ============================================================================
    print("Generating Section 4: Match Statistics...")
    
    # 4.1 Distribution of Match Stats (Histograms)
    stats_cols = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF']
    stats_names = ['Home Shots', 'Away Shots', 'Home Shots on Target', 
                   'Away Shots on Target', 'Home Corners', 'Away Corners', 
                   'Home Fouls', 'Away Fouls']
    
    for i in range(0, len(stats_cols), 4):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f"4.{i//4 + 1} Match Statistics Distribution", 
                     fontsize=16, fontweight='bold')
        
        for j in range(4):
            if i + j < len(stats_cols):
                col = stats_cols[i + j]
                name = stats_names[i + j]
                ax = axes[j // 2, j % 2]
                
                ax.hist(df[col].dropna(), bins=20, color='steelblue', 
                        edgecolor='black', alpha=0.7)
                ax.set_title(name, fontweight='bold')
                ax.set_xlabel(name, fontsize=10)
                ax.set_ylabel("Frequency", fontsize=10)
                ax.axvline(df[col].mean(), color='red', linestyle='--', 
                           label=f'Mean: {df[col].mean():.2f}')
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # 4.2 Home Advantage Analysis (Box Plots)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("4.3 Home Advantage Analysis: Home vs Away Statistics", 
                 fontsize=16, fontweight='bold')
    
    # Shots
    shots_data = pd.DataFrame({
        'Shots': list(df['HS'].dropna()) + list(df['AS'].dropna()),
        'Location': ['Home']*len(df['HS'].dropna()) + ['Away']*len(df['AS'].dropna())
    })
    sns.boxplot(data=shots_data, x='Location', y='Shots', 
                palette=['#4472C4', '#FF5733'], ax=axes[0, 0])
    axes[0, 0].set_title("Shots", fontweight='bold')
    axes[0, 0].set_ylabel("Number of Shots", fontsize=10)
    
    # Shots on Target
    sot_data = pd.DataFrame({
        'Shots on Target': list(df['HST'].dropna()) + list(df['AST'].dropna()),
        'Location': ['Home']*len(df['HST'].dropna()) + ['Away']*len(df['AST'].dropna())
    })
    sns.boxplot(data=sot_data, x='Location', y='Shots on Target', 
                palette=['#4472C4', '#FF5733'], ax=axes[0, 1])
    axes[0, 1].set_title("Shots on Target", fontweight='bold')
    axes[0, 1].set_ylabel("Number of Shots on Target", fontsize=10)
    
    # Corners
    corners_data = pd.DataFrame({
        'Corners': list(df['HC'].dropna()) + list(df['AC'].dropna()),
        'Location': ['Home']*len(df['HC'].dropna()) + ['Away']*len(df['AC'].dropna())
    })
    sns.boxplot(data=corners_data, x='Location', y='Corners', 
                palette=['#70AD47', '#C55A11'], ax=axes[1, 0])
    axes[1, 0].set_title("Corners", fontweight='bold')
    axes[1, 0].set_ylabel("Number of Corners", fontsize=10)
    
    # Fouls
    fouls_data = pd.DataFrame({
        'Fouls': list(df['HF'].dropna()) + list(df['AF'].dropna()),
        'Location': ['Home']*len(df['HF'].dropna()) + ['Away']*len(df['AF'].dropna())
    })
    sns.boxplot(data=fouls_data, x='Location', y='Fouls', 
                palette=['#FFC000', '#C55A11'], ax=axes[1, 1])
    axes[1, 1].set_title("Fouls", fontweight='bold')
    axes[1, 1].set_ylabel("Number of Fouls", fontsize=10)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 4.4 Disciplinary Analysis
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("4.4 Disciplinary Analysis", fontsize=16, fontweight='bold')
    
    # Home Yellow Cards
    axes[0, 0].hist(df['HY'].dropna(), bins=range(0, int(df['HY'].max())+2), 
                    color='#FFC000', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title("Home Yellow Cards", fontweight='bold')
    axes[0, 0].set_xlabel("Yellow Cards", fontsize=10)
    axes[0, 0].set_ylabel("Frequency", fontsize=10)
    
    # Away Yellow Cards
    axes[0, 1].hist(df['AY'].dropna(), bins=range(0, int(df['AY'].max())+2), 
                    color='#FFC000', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("Away Yellow Cards", fontweight='bold')
    axes[0, 1].set_xlabel("Yellow Cards", fontsize=10)
    axes[0, 1].set_ylabel("Frequency", fontsize=10)
    
    # Home Red Cards
    axes[1, 0].hist(df['HR'].dropna(), bins=range(0, int(df['HR'].max())+2), 
                    color='#FF0000', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Home Red Cards", fontweight='bold')
    axes[1, 0].set_xlabel("Red Cards", fontsize=10)
    axes[1, 0].set_ylabel("Frequency", fontsize=10)
    
    # Away Red Cards
    axes[1, 1].hist(df['AR'].dropna(), bins=range(0, int(df['AR'].max())+2), 
                    color='#FF0000', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Away Red Cards", fontweight='bold')
    axes[1, 1].set_xlabel("Red Cards", fontsize=10)
    axes[1, 1].set_ylabel("Frequency", fontsize=10)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # 4.5 Red Cards by Team
    home_reds = df.groupby('HomeTeam')['HR'].sum()
    away_reds = df.groupby('AwayTeam')['AR'].sum()
    total_reds = (home_reds.add(away_reds, fill_value=0)).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    total_reds.plot(kind='bar', ax=ax, color='red', alpha=0.7)
    ax.set_title("4.5 Total Red Cards Issued per Team", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Team", fontsize=12)
    ax.set_ylabel("Total Red Cards", fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # SECTION 5: CORRELATION & KEY DRIVER ANALYSIS
    # ============================================================================
    print("Generating Section 5: Correlation Analysis...")
    
    # 5.1 Match Stats Correlation Heatmap
    corr_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 
                 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    
    # Filter out columns that might not be present
    corr_cols = [col for col in corr_cols if col in df.columns]
    
    corr_matrix = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(11, 9)) # Make it slightly larger
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                linewidths=0.5, annot_kws={"size": 8}, ax=ax)
    ax.set_title("5.1 Match Statistics Correlation Heatmap", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 5.2 Shots vs. Goals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle("5.2 Shots on Target vs. Goals", fontsize=16, fontweight='bold')
    
    # Home Shots vs Goals
    sns.regplot(data=df, x='HST', y='FTHG', ax=ax1, 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    ax1.set_title("Home Shots on Target vs. Home Goals", fontweight='bold')
    ax1.set_xlabel("Home Shots on Target (HST)", fontsize=12)
    ax1.set_ylabel("Full-Time Home Goals (FTHG)", fontsize=12)
    
    # Away Shots vs Goals
    sns.regplot(data=df, x='AST', y='FTAG', ax=ax2, 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    ax2.set_title("Away Shots on Target vs. Away Goals", fontweight='bold')
    ax2.set_xlabel("Away Shots on Target (AST)", fontsize=12)
    ax2.set_ylabel("Full-Time Away Goals (FTAG)", fontsize=12)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 5.3 Corners vs. Goals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle("5.3 Corners vs. Goals", fontsize=16, fontweight='bold')
    
    # Home Corners vs Goals (using lowess for a non-linear trend)
    sns.regplot(data=df, x='HC', y='FTHG', ax=ax1, 
                scatter_kws={'alpha':0.3}, line_kws={'color':'blue'}, lowess=True)
    ax1.set_title("Home Corners vs. Home Goals", fontweight='bold')
    ax1.set_xlabel("Home Corners (HC)", fontsize=12)
    ax1.set_ylabel("Full-Time Home Goals (FTHG)", fontsize=12)
    
    # Away Corners vs Goals
    sns.regplot(data=df, x='AC', y='FTAG', ax=ax2, 
                scatter_kws={'alpha':0.3}, line_kws={'color':'blue'}, lowess=True)
    ax2.set_title("Away Corners vs. Away Goals", fontweight='bold')
    ax2.set_xlabel("Away Corners (AC)", fontsize=12)
    ax2.set_ylabel("Full-Time Away Goals (FTAG)", fontsize=12)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ============================================================================
    # SECTION 6: REFEREE ANALYSIS
    # ============================================================================
    print("Generating Section 6: Referee Analysis...")
    
    if 'Referee' in df.columns:
        # 6.1 Referee Match Load
        ref_stats = df.groupby('Referee').size().reset_index(name='Matches')
        top_15_refs = ref_stats.sort_values('Matches', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        sns.barplot(data=top_15_refs, x='Matches', y='Referee', orient='h', 
                    palette='Blues_r', ax=ax)
        ax.set_title("6.1 Top 15 Referees by Matches Officiated", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Number of Matches", fontsize=12)
        ax.set_ylabel("Referee", fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 6.2 Referee Disciplinary Record
        ref_discipline = df.groupby('Referee').agg({
            'HY': 'sum', 'AY': 'sum', 'HR': 'sum', 'AR': 'sum'
        }).reset_index()
        ref_discipline['TotalYellows'] = ref_discipline['HY'] + ref_discipline['AY']
        ref_discipline['TotalReds'] = ref_discipline['HR'] + ref_discipline['AR']
        
        ref_stats = pd.merge(ref_stats, ref_discipline, on='Referee')
        ref_stats['AvgYellowsPerGame'] = ref_stats['TotalYellows'] / ref_stats['Matches']
        ref_stats['AvgRedsPerGame'] = ref_stats['TotalReds'] / ref_stats['Matches']
        
        # Get stats for the same top 15 referees
        top_ref_stats = ref_stats[ref_stats['Referee'].isin(top_15_refs['Referee'])]
        
        # Avg Yellows
        fig, ax = plt.subplots(figsize=(11, 8.5))
        sns.barplot(data=top_ref_stats.sort_values('AvgYellowsPerGame', ascending=False),
                    x='AvgYellowsPerGame', y='Referee', orient='h', palette='YlOrBr_r', ax=ax)
        ax.set_title("6.2 Average Yellow Cards per Game (Top 15 Referees)", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Average Yellow Cards", fontsize=12)
        ax.set_ylabel("Referee", fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Total Reds
        fig, ax = plt.subplots(figsize=(11, 8.5))
        sns.barplot(data=top_ref_stats.sort_values('TotalReds', ascending=False),
                    x='TotalReds', y='Referee', orient='h', palette='Reds_r', ax=ax)
        ax.set_title("6.3 Total Red Cards Issued (Top 15 Referees)", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Total Red Cards", fontsize=12)
        ax.set_ylabel("Referee", fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    else:
        add_text_page(pdf, "Section 6: Referee Analysis", "No 'Referee' column found in the dataset.")

    # ============================================================================
    # SECTION 7: BETTING ODDS ANALYSIS
    # ============================================================================
    print("Generating Section 7: Betting Odds Analysis...")
    
    bet_cols = ['B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5']
    if all(col in df.columns for col in bet_cols):
        # 7.1 Bookmaker Odds vs. Actual Results
        fig, axes = plt.subplots(1, 3, figsize=(11, 8.5))
        fig.suptitle("7.1 Bet365 Odds vs. Full-Time Result", fontsize=16, fontweight='bold')
        
        sns.boxplot(data=df, x='FTR', y='B365H', ax=axes[0], palette=['#4472C4', '#FFC000', '#FF5733'])
        axes[0].set_title("Home Win Odds (B365H)", fontweight='bold')
        axes[0].set_xlabel("Full-Time Result", fontsize=10)
        axes[0].set_ylabel("Betting Odds", fontsize=10)
        
        sns.boxplot(data=df, x='FTR', y='B365D', ax=axes[1], palette=['#4472C4', '#FFC000', '#FF5733'])
        axes[1].set_title("Draw Odds (B365D)", fontweight='bold')
        axes[1].set_xlabel("Full-Time Result", fontsize=10)
        axes[1].set_ylabel("Betting Odds", fontsize=10)
        
        sns.boxplot(data=df, x='FTR', y='B365A', ax=axes[2], palette=['#4472C4', '#FFC000', '#FF5733'])
        axes[2].set_title("Away Win Odds (B365A)", fontweight='bold')
        axes[2].set_xlabel("Full-Time Result", fontsize=10)
        axes[2].set_ylabel("Betting Odds", fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 7.2 Over/Under 2.5 Goals Analysis
        df['Over2.5'] = df['TotalGoals'] > 2.5
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        sns.boxplot(data=df, x='Over2.5', y='B365>2.5', palette=['#FF5733', '#4472C4'], ax=ax)
        ax.set_title("7.2 Bet365 Over 2.5 Odds vs. Actual Outcome", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Actual Result (True = Over 2.5 Goals)", fontsize=12)
        ax.set_ylabel("B365 Over 2.5 Odds", fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 7.3 Over/Under Accuracy Table
        df['BookiePredOver2.5'] = df['B365>2.5'] < df['B365<2.5']
        cm = pd.crosstab(df['Over2.5'], df['BookiePredOver2.5'])
        cm.index.name = 'Actual Over 2.5'
        cm.columns.name = 'Bookie Predicted Over 2.5'
        add_table_to_pdf(pdf, cm, "7.3 Bookmaker Prediction Accuracy (B365>2.5 < B365<2.5)")
        
        # 7.4 Market Average Correlation
        avg_cols = ['AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']
        outcome_cols = ['HomeWin', 'DrawResult', 'AwayWin', 'Over2.5']
        
        if all(col in df.columns for col in avg_cols):
            corr_data = df[avg_cols + outcome_cols]
            corr_matrix = corr_data.corr()
            corr_subset = corr_matrix.loc[avg_cols, outcome_cols]
            
            fig, ax = plt.subplots(figsize=(11, 8.5))
            sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='viridis', 
                        linewidths=0.5, ax=ax)
            ax.set_title("7.4 Market Average Odds vs. Actual Outcomes Correlation", 
                         fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Actual Outcome", fontsize=12)
            ax.set_ylabel("Average Market Odd", fontsize=12)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        else:
            add_text_page(pdf, "Section 7.4: Market Average Correlation", "Average odds columns (AvgH, AvgD, etc.) not found.")
            
    else:
        add_text_page(pdf, "Section 7: Betting Odds Analysis", "Betting columns (B365H, B365>2.5, etc.) not found.")

    # ============================================================================
    # SECTION 8: TEMPORAL ANALYSIS
    # ============================================================================
    print("Generating Section 8: Temporal Analysis...")
    
    # Sort by date for temporal plots
    df = df.sort_values('Date')
    
    # 8.1 Goals Over Time
    # Using a 38-game rolling average (approx 10% of a season, ~3-4 matchdays)
    df['TotalGoals_RollAvg'] = df['TotalGoals'].rolling(window=38, center=True, min_periods=10).mean()
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.plot(df['Date'], df['TotalGoals'], color='gray', alpha=0.2, label='Total Goals (Raw)')
    ax.plot(df['Date'], df['TotalGoals_RollAvg'], color='blue', 
            linewidth=2, label='Rolling Average (38-game)')
    ax.set_title("8.1 Total Goals per Match Over Time", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Total Goals", fontsize=12)
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 8.2 Cards Over Time
    df['TotalYellows'] = df['HY'].fillna(0) + df['AY'].fillna(0)
    df['TotalYellows_RollAvg'] = df['TotalYellows'].rolling(window=38, center=True, min_periods=10).mean()
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.plot(df['Date'], df['TotalYellows'], color='gray', alpha=0.2, label='Total Yellow Cards (Raw)')
    ax.plot(df['Date'], df['TotalYellows_RollAvg'], color='orange', 
            linewidth=2, label='Rolling Average (38-game)')
    ax.set_title("8.2 Total Yellow Cards per Match Over Time", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Total Yellow Cards", fontsize=12)
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # 8.3 Results by Month
    month_stats = df.groupby(['Month', 'MonthName'])[['HomeWin', 'DrawResult', 'AwayWin']].sum().reset_index()
    month_stats = month_stats.sort_values('Month')
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    month_stats.plot(kind='bar', x='MonthName', y=['HomeWin', 'DrawResult', 'AwayWin'], 
                     color=['#4472C4', '#FFC000', '#FF5733'], ax=ax)
    ax.set_title("8.3 Match Results by Month", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Number of Matches", fontsize=12)
    ax.legend(['Home Wins', 'Draws', 'Away Wins'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# End of with PdfPages block
print("==========================================================")
print("PDF generation complete! File saved as E0_EDA_Report.pdf")
print("==========================================================")