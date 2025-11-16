import pandas as pd
import numpy as np

class DataProvider:
    """
    Provides time-based match features and events for prediction generation.
    Reads from a master Premier_League.csv file.
    """

    def __init__(self, csv_path):
        # Load your dataset
        self.df = pd.read_csv(csv_path)
        # Normalize column names (strip spaces, lower)
        self.df.columns = [c.strip().lower() for c in self.df.columns]

    def features_at_time(self, t, home, away):
        """
        Return feature vector for time 't' seconds in match between home and away.
        Feature examples:
         - possession difference
         - shots difference
         - fouls
         - corners
         - expected goals (xG)
        """
        # Convert time in seconds to match minute
        minute = int(t / 60)

        # Filter for this specific match
        match_data = self.df[
            (self.df["hometeam"].str.lower() == home.lower()) &
            (self.df["awayteam"].str.lower() == away.lower())
        ]

        if match_data.empty:
            # If no exact match found, return neutral features
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        # Find nearest minute data point if dataset is time-granular
        if "minute" in match_data.columns:
            nearest = match_data.iloc[(match_data["minute"] - minute).abs().argsort()[:1]]
        else:
            nearest = match_data.iloc[:1]

        row = nearest.iloc[0]

        # Choose 5 key features — adjust based on your actual CSV columns
        possession_diff = row.get("homepossession", 50) - row.get("awaypossession", 50)
        shots_diff = row.get("homeshots", 0) - row.get("awayshots", 0)
        xg_diff = row.get("homexg", 0) - row.get("awayxg", 0)
        fouls_diff = row.get("homefouls", 0) - row.get("awayfouls", 0)
        corners_diff = row.get("homecorners", 0) - row.get("awaycorners", 0)

        return [possession_diff, shots_diff, xg_diff, fouls_diff, corners_diff]

    def match_events(self, home, away):
        """
        Extract actual match events (goals, cards, substitutions) for slider demo.
        Returns a list of dicts: [{"time": 1620, "type": "goal", "team": "Arsenal"}]
        """
        match_data = self.df[
            (self.df["hometeam"].str.lower() == home.lower()) &
            (self.df["awayteam"].str.lower() == away.lower())
        ]

        events = []

        if "eventtype" in match_data.columns:
            for _, row in match_data.iterrows():
                event_type = str(row["eventtype"]).lower()
                if event_type in ["goal", "red card", "yellow card", "substitution"]:
                    events.append({
                        "time": int(row.get("minute", 0)) * 60,
                        "type": event_type,
                        "team": row.get("team", home if "home" in event_type else away),
                        "desc": row.get("eventdesc", event_type.capitalize())
                    })

        # Fallback — if no events found
        if not events:
            events.append({"time": 1620, "type": "goal", "team": away, "desc": "Goal by away team"})

        return events
