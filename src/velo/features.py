import numpy as np
import pandas as pd

from velo import context
from velo.config import MIN_BIKES

HORIZONS = (15, 30, 60)  # minutes

FEATURES = [
    "bikes", "docks", "capacity", "fill",
    "d15", "d60",
    # pas de férié en sept-oct donc pas de colonne public_holiday, ça passe par le profil
    "slot", "dow", "weekend", "school_holiday",
    "rain", "temp",
    "profile",
]


def day_type(df):
    return np.where(df["weekend"] | df["public_holiday"], "off", "work")


def add_calendar(df, weather):
    df = pd.concat([df, context.day_flags(df["ts"])], axis=1)
    df["slot"] = (df["ts"].dt.hour * 4 + df["ts"].dt.minute // 15).astype("int16")
    df["dow"] = df["ts"].dt.dayofweek.astype("int8")
    # météo de la dernière heure finie (l'heure en cours n'est pas finie à t).
    # floor en UTC sinon ça plante sur l'heure ambiguë du changement d'heure
    last_hour = df["ts"].dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(df["ts"].dt.tz) - pd.Timedelta("1h")
    df["rain"] = last_hour.map(weather["rain"]).astype("float32")
    df["temp"] = last_hour.map(weather["temp"]).astype("float32")
    df["day_type"] = day_type(df)
    return df


def station_features(st):
    # une seule station, grille régulière
    st = st.sort_values("ts")
    bikes = st["bikes"]
    res = st.copy()
    res["fill"] = (bikes / st["capacity"]).astype("float32")
    res["d15"] = bikes - bikes.shift(1)
    res["d60"] = bikes - bikes.shift(4)
    for h in HORIZONS:
        future = bikes.shift(-h // 15)
        res[f"y{h}"] = (future >= MIN_BIKES).where(future.notna())
    return res


def profile_table(df):
    # P(>= 2 vélos) par station / type de jour / quart d'heure. c'est la baseline "profil horaire"
    ok = (df["bikes"] >= MIN_BIKES).where(df["bikes"].notna())
    return (ok.groupby([df["station"], df["day_type"], df["slot"]]).mean()
            .rename("profile").astype("float32"))


def add_profile(df, table):
    idx = pd.MultiIndex.from_frame(df[["station", "day_type", "slot"]])
    df["profile"] = table.reindex(idx).to_numpy()
    return df
