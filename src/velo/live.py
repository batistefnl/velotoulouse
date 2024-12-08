# prédictions à partir du flux temps réel, pour l'app

import joblib
import pandas as pd
import requests

from velo import features as F
from velo.config import MODELS, TZ
from velo.model import BASE

GBFS_URL = "https://api.cyclocity.fr/contracts/toulouse/gbfs/v2/"


def fetch_status():
    info = requests.get(GBFS_URL + "station_information.json", timeout=10)
    status = requests.get(GBFS_URL + "station_status.json", timeout=10)
    info.raise_for_status()
    status.raise_for_status()
    info, status = info.json()["data"]["stations"], status.json()["data"]["stations"]
    df = pd.DataFrame(info)[["station_id", "name", "lat", "lon", "capacity"]].merge(
        pd.DataFrame(status)[["station_id", "num_bikes_available", "num_docks_available", "is_installed",
                              "is_renting"]])
    df = df.rename(columns={"station_id": "station", "num_bikes_available": "bikes",
                            "num_docks_available": "docks"})
    df["station"] = df["station"].astype(int)
    # comme dans le collecteur. parfois des booléens, parfois 0/1...
    df["open"] = df["is_installed"].astype(bool) & df["is_renting"].astype(bool)
    return df


def fetch_weather(now):
    # météo-france c'est pas en temps réel, donc open-meteo à Blagnac à la place.
    # pluie cumulée sur la dernière heure finie, temp à la fin de l'heure, comme à l'entrainement
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
        "latitude": 43.6211, "longitude": 1.3788, "timezone": "GMT",
        "hourly": "temperature_2m,precipitation", "past_hours": 3, "forecast_hours": 1})
    r.raise_for_status()
    q = r.json()["hourly"]
    # open-meteo date l'heure par sa fin aussi. si la dernière heure est pas encore publiée
    # on prend la plus récente dispo
    times = pd.to_datetime(q["time"], utc=True)
    known = [i for i, t in enumerate(times)
             if t <= now.tz_convert("UTC").floor("h") and q["precipitation"][i] is not None]
    i = known[-1]
    end = times[i]
    return pd.DataFrame({"rain": [q["precipitation"][i]], "temp": [q["temperature_2m"][i]]},
                        index=pd.DatetimeIndex([(end - pd.Timedelta("1h")).tz_convert(TZ)]))


def load_models():
    models = {h: joblib.load(MODELS / f"gbm_{h}.joblib") for h in F.HORIZONS}
    return models, joblib.load(MODELS / "profile.joblib")


def predict(status, now, weather, models, profile):
    df = status.assign(ts=now.tz_convert(TZ))
    df = F.add_calendar(df, weather)
    df["fill"] = df["bikes"] / df["capacity"]
    for c in ("weekend", "public_holiday", "school_holiday"):
        df[c] = df[c].astype("int8")
    df = F.add_profile(df, profile)
    for h, model in models.items():
        df[f"p{h}"] = model.predict_proba(df[BASE])[:, 1]
        # le modèle n'a jamais vu de station fermée (manquant à l'entrainement)
        df.loc[~df["open"], f"p{h}"] = float("nan")
    return df
