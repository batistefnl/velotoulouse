from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"  # y copier les collected_*.parquet du collecteur
PROCESSED = DATA / "processed"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

TZ = "Europe/Paris"

# sept-nov 2024 (premier relevé le 1er sept à 2h30). train sept+oct, test nov
START = pd.Timestamp("2024-09-01", tz=TZ)
END = pd.Timestamp("2024-12-01", tz=TZ)
TEST_START = pd.Timestamp("2024-11-01", tz=TZ)

MIN_BIKES = 2  # 1 vélo affiché c'est souvent un vélo cassé

# mon trajet. HOME_COORDS c'est à peu près le quartier, pas l'adresse
HOME_COORDS = (43.566, 1.454)
SCHOOL_COORDS = (43.5683, 1.4721)
HOME = {358: "Salade Ponsan - Côteaux", 359: "Salade Ponsan - Sahuque",
        232: "Narbonne - Caubère", 233: "Narbonne - Sahuque"}
SCHOOL = {225: "Belin - Onera", 224: "Belin - Supaero", 230: "UT3 - Champs Magnétiques"}
STATIONS = {**HOME, **SCHOOL}
