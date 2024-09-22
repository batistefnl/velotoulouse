import pandas as pd
import requests

from velo.config import END, RAW, START, TZ

METEO_URL = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR/{}"
METEO_FILE = "H_31_latest-2023-2024.csv.gz"
BLAGNAC = 31069001

SCHOOL_URL = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/"
    "exports/json?where=location%3D%22Toulouse%22"
)
PUBLIC_URL = "https://calendrier.api.gouv.fr/jours-feries/metropole/{}.json"


def _get(url, path, refresh=False):
    if refresh or not path.exists():
        print(f"downloading {url}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def weather(start=START, end=END):
    """Pluie (mm) et température horaires à Blagnac, indexées par le début de l'heure en local."""
    path = _get(METEO_URL.format(METEO_FILE), RAW / METEO_FILE)
    df = pd.read_csv(path, sep=";", usecols=["NUM_POSTE", "AAAAMMJJHH", "RR1", "T"])
    df = df[df["NUM_POSTE"] == BLAGNAC]
    # météo-france date l'heure par sa fin, en UTC
    ts = pd.to_datetime(df["AAAAMMJJHH"].astype(str), format="%Y%m%d%H", utc=True) - pd.Timedelta("1h")
    res = (pd.DataFrame({"rain": df["RR1"].to_numpy(), "temp": df["T"].to_numpy()},
                        index=ts.dt.tz_convert(TZ).to_numpy())
           .sort_index())
    res = res[~res.index.duplicated()]
    # to_numpy perd le fuseau, on le remet ici. pas très propre mais ça marche
    res.index = pd.DatetimeIndex(res.index).tz_convert(TZ)
    res.index.name = "hour"
    return res.loc[start:end - pd.Timedelta("1s")]


def school_holidays():
    df = pd.read_json(_get(SCHOOL_URL, RAW / "calendrier_scolaire_toulouse.json"))
    df = df[df["population"].isin(["-", "Élèves"])]
    # minuit heure locale stocké en UTC. la fin = le matin de la rentrée
    start = pd.to_datetime(df["start_date"], utc=True).dt.tz_convert(TZ).dt.normalize()
    end = pd.to_datetime(df["end_date"], utc=True).dt.tz_convert(TZ).dt.normalize()
    return list(zip(start, end))


def public_holidays(years=None):
    days = []
    for year in years or range(START.year, END.year + 1):
        path = _get(PUBLIC_URL.format(year), RAW / f"jours_feries_{year}.json")
        days += list(pd.read_json(path, typ="series").index)
    return pd.DatetimeIndex(days).tz_localize(TZ)


def day_flags(ts):
    day = ts.dt.normalize()
    school = pd.Series(False, index=ts.index)
    for start, end in school_holidays():
        school |= (day >= start) & (day < end)
    return pd.DataFrame({
        "weekend": ts.dt.dayofweek >= 5,
        "public_holiday": day.isin(public_holidays()),
        "school_holiday": school,
    }, index=ts.index)
