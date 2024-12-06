# P(au moins 2 vélos dans h minutes). train sept-oct 2024, test novembre.
# seul truc décidé en regardant le test : virer d15/d60, ça change presque rien

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from velo import context
from velo import features as F
from velo.config import MODELS, PROCESSED, STATIONS, TEST_START
from velo.data import load_grid

DATASET_FILE = PROCESSED / "dataset.parquet"
METRICS_FILE = PROCESSED / "metrics.csv"
CALIBRATION_FILE = PROCESSED / "calibration.csv"

BASE = [c for c in F.FEATURES if c not in ("d15", "d60")]  # ce que donne le flux temps réel
TREND = BASE + ["d15", "d60"]
LINEAR = ["bikes", "docks", "fill", "profile", "rain", "temp", "weekend", "school_holiday"]


def build_dataset():
    grid = F.add_calendar(load_grid(), context.weather())
    df = pd.concat([F.station_features(st) for _, st in grid.groupby("station", sort=False)],
                   ignore_index=True)
    for c in ("weekend", "public_holiday", "school_holiday"):
        df[c] = df[c].astype("int8")
    df.to_parquet(DATASET_FILE)
    return df


def split(df):
    train, test = df[df["ts"] < TEST_START].copy(), df[df["ts"] >= TEST_START].copy()
    # sinon les dernières lignes d'octobre ont leur cible en novembre
    for h in F.HORIZONS:
        train.loc[train["ts"] + pd.Timedelta(minutes=h) >= TEST_START, f"y{h}"] = np.nan
    table = F.profile_table(train)
    return F.add_profile(train, table), F.add_profile(test, table), table


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          random_state=0)


def scores(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, p), "auc": roc_auc_score(y, p)}


def evaluate(df):
    train, test, table = split(df)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(table, MODELS / "profile.joblib")
    rows, calib = [], []
    for h in F.HORIZONS:
        y = f"y{h}"
        a = train.dropna(subset=TREND + [y])  # mêmes lignes pour tous les modèles
        b = test.dropna(subset=TREND + [y])
        ya, yb = a[y].astype(int), b[y].astype(int)

        # persistance = P(y | nb de vélos maintenant), connait l'état mais pas l'heure
        persistence = ya.groupby(a["bikes"].clip(upper=15)).mean()
        linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        linear.fit(a[LINEAR], ya)
        model = gbm().fit(a[BASE], ya)
        trend = gbm().fit(a[TREND], ya)
        # model = gbm().fit(a[TREND], ya)
        joblib.dump(model, MODELS / f"gbm_{h}.joblib")

        preds = {
            "profil horaire": b["profile"].to_numpy(),
            "persistance": b["bikes"].clip(upper=15).map(persistence).to_numpy(),
            "régression logistique": linear.predict_proba(b[LINEAR])[:, 1],
            "gradient boosting": model.predict_proba(b[BASE])[:, 1],
            "gradient boosting + tendance": trend.predict_proba(b[TREND])[:, 1],
        }
        subsets = {
            "réseau": np.ones(len(b), bool),
            "mes 7 stations": b["station"].isin(list(STATIONS)).to_numpy(),
            "1 à 4 vélos": b["bikes"].between(1, 4).to_numpy(),
        }
        for subset, mask in subsets.items():
            for name, p in preds.items():
                rows.append({"horizon": h, "subset": subset, "model": name, "n": int(mask.sum()),
                             **scores(yb[mask], p[mask])})

        frac, mean = calibration_curve(yb, preds["gradient boosting"], n_bins=10, strategy="quantile")
        calib.append(pd.DataFrame({"horizon": h, "predicted": mean, "observed": frac}))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_FILE, index=False)
    pd.concat(calib).to_csv(CALIBRATION_FILE, index=False)
    return metrics


if __name__ == "__main__":
    df = build_dataset()
    m = evaluate(df)
    with pd.option_context("display.width", 200):
        print(m.round(4).to_string(index=False))
