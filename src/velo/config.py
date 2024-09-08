from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"  # y copier les collected_*.parquet du collecteur
PROCESSED = DATA / "processed"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

TZ = "Europe/Paris"

# premier relevé le 1er sept à 2h30
START = pd.Timestamp("2024-09-01", tz=TZ)
END = pd.Timestamp("2024-10-01", tz=TZ)

MIN_BIKES = 2  # 1 vélo affiché c'est souvent un vélo cassé

# mon trajet
HOME = {358: "Salade Ponsan - Côteaux", 359: "Salade Ponsan - Sahuque",
        232: "Narbonne - Caubère", 233: "Narbonne - Sahuque"}
SCHOOL = {225: "Belin - Onera", 224: "Belin - Supaero", 230: "UT3 - Champs Magnétiques"}
STATIONS = {**HOME, **SCHOOL}
