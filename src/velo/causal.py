# effet de la pluie sur l'usage du réseau.
# activité = variation absolue moyenne du nb de vélos par station et par quart d'heure
# (pas exactement les locations : 2 mouvements dans le même 1/4h s'annulent, + les camions).
# effets fixes date + heure (par type de jour) -> on compare les heures de pluie aux heures
# sèches de la même journée

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from velo import context
from velo.config import TZ
from velo.data import GRID_FILE
from velo.features import day_type

DAY_HOURS = range(7, 22)  # en dehors le réseau dort
WET = 0.5  # mm dans l'heure. la plupart des valeurs sont des multiples de 0.2


def hourly_activity():
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    df = con.execute(f"""
        WITH moves AS (
            SELECT station, ts,
                   abs(bikes - lag(bikes) OVER (PARTITION BY station ORDER BY ts)) AS move
            FROM (SELECT station, ts, CASE WHEN open THEN bikes END AS bikes FROM '{GRID_FILE}')
        )
        -- le mouvement à ts a eu lieu dans le quart d'heure d'avant
        SELECT date_trunc('hour', ts - INTERVAL '15 minutes') AS hour, avg(move) AS activity,
               count(move) AS n
        FROM moves WHERE move IS NOT NULL GROUP BY 1 ORDER BY 1
    """).df()
    df["hour"] = df["hour"].dt.tz_convert(TZ)
    return df


def panel(min_obs=1000):
    df = hourly_activity()
    # heures presque vides dans les relevés, et heures où le flux était figé (0 mouvement
    # sur tout le réseau en pleine journée)
    df = df[(df["n"] >= min_obs) & (df["activity"] > 0)]
    df = df.merge(context.weather(), left_on="hour", right_index=True)
    df = pd.concat([df.reset_index(drop=True), context.day_flags(df["hour"]).reset_index(drop=True)], axis=1)
    df["date"] = df["hour"].dt.strftime("%Y-%m-%d")
    df["h"] = df["hour"].dt.hour
    df["day_type"] = day_type(df)
    df["wet"] = (df["rain"] >= WET).astype(float)
    # la bruine à part pour qu'elle soit pas dans les heures "sèches"
    df["drizzle"] = ((df["rain"] > 0) & (df["rain"] < WET)).astype(float)
    df = df.sort_values("hour").reset_index(drop=True)
    return df[df["h"].isin(DAY_HOURS)].reset_index(drop=True)


def fit(df, treatment, outcome="log_activity"):
    """MCO avec température, effets fixes date et heure, erreurs groupées par date."""
    df = df.dropna(subset=list(treatment) + ["temp"])
    y = np.log(df["activity"]) if outcome == "log_activity" else df[outcome]

    dates = pd.get_dummies(df["date"], prefix="d", dtype=float)
    tmp = {}
    for dt in ("work", "off"):
        for h in DAY_HOURS:
            if h != 12:  # heure de référence
                tmp[f"h{h}_{dt}"] = ((df["h"] == h) & (df["day_type"] == dt)).astype(float)
    hours = pd.DataFrame(tmp, index=df.index)
    hours = hours.loc[:, hours.any()]  # un type de jour peut manquer

    X = pd.concat([df[list(treatment)], df["temp"], (df["temp"] ** 2).rename("temp2"), dates, hours], axis=1)
    # groupé par date parce que les heures de pluie viennent en série
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["date"])[0]})
    out = pd.DataFrame({"coef": res.params, "se": res.bse}).loc[list(treatment)]
    out["low"] = out["coef"] - 1.96 * out["se"]
    out["high"] = out["coef"] + 1.96 * out["se"]
    out.attrs["nobs"] = int(res.nobs)
    return out


def as_percent(table):
    return 100 * np.expm1(table[["coef", "low", "high"]])


if __name__ == "__main__":
    df = panel()
    print(f"{len(df)} hours, {int(df['wet'].sum())} wet, {df['date'].nunique()} days\n")
    print("Wet hour (>= 0.5 mm), % change in activity:")
    print(as_percent(fit(df, ["wet", "drizzle"])).round(1).to_string(), "\n")

    df2 = df.assign(moderate=((df["rain"] >= WET) & (df["rain"] <= 1)).astype(float),
                    heavy=(df["rain"] > 1).astype(float))
    print(as_percent(fit(df2, ["drizzle", "moderate", "heavy"])).round(1).to_string())
