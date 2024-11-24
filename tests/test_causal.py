import numpy as np
import pandas as pd

from velo import causal


def synthetic_panel(effect=-0.2, days=60, seed=0):
    # jours de pluie plus calmes toute la journée + vrai effet de -0.2 dans l'heure
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2024-09-02", periods=days, freq="D", tz="Europe/Paris"):
        rainy_day = rng.random() < 0.4
        day_level = -0.3 if rainy_day else 0.0
        for h in causal.DAY_HOURS:
            wet = float(rainy_day and rng.random() < 0.3)
            cycle = 0.5 * np.sin(np.pi * (h - 7) / 14)
            log_act = cycle + day_level + effect * wet + rng.normal(0, 0.05)
            rows.append({"hour": d + pd.Timedelta(hours=h), "date": d.strftime("%Y-%m-%d"), "h": h,
                         "day_type": "work", "wet": wet, "temp": 15.0 + rng.normal(),
                         "activity": np.exp(log_act)})
    return pd.DataFrame(rows)


def test_fixed_effects_recover_the_within_day_effect():
    df = synthetic_panel()
    est = causal.fit(df, ["wet"]).loc["wet"]
    assert est["low"] < -0.2 < est["high"]
    # sans effets fixes date on surestime l'effet
    naive = np.log(df.loc[df.wet == 1, "activity"]).mean() - np.log(df.loc[df.wet == 0, "activity"]).mean()
    assert naive < -0.3


def test_as_percent():
    t = pd.DataFrame({"coef": [np.log(0.8)], "low": [0.0], "high": [0.0]})
    assert round(causal.as_percent(t)["coef"].iloc[0], 6) == -20
