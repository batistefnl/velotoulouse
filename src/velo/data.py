import duckdb
import pandas as pd

from velo.config import END, PROCESSED, RAW, START, TZ

GRID_FILE = PROCESSED / "grid.parquet"
STATIONS_FILE = PROCESSED / "stations.parquet"

# le collecteur tourne au quart d'heure mais enregistre qq secondes après (10:15:02),
# ce relevé c'est l'état à 10:15 et pas 10:30
LATE = "1 minute"


def _collected():
    if not any(RAW.glob("collected_*.parquet")):
        raise FileNotFoundError(f"no collected_*.parquet in {RAW}: copy the collector's data/ there first")
    return f"(SELECT *, collected_at_utc AS at_utc FROM read_parquet('{RAW}/collected_*.parquet'))"


def build(start=START, end=END, tolerance="30 minutes"):
    """Construit stations.parquet et la grille de 15 min dans data/processed."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute("SET preserve_insertion_order=false")
    t0, t1 = start.tz_convert("UTC"), end.tz_convert("UTC")

    # les noms changent (accents, rues renommées) mais pas le numéro devant -> c'est l'id.
    # any_value au cas où il y aurait des doublons (station, relevé), j'en ai pas vu
    con.execute(f"""
        CREATE OR REPLACE TABLE snap AS
        SELECT
            station, at_utc AS ts,
            any_value(num_bikes_available) AS bikes,
            any_value(num_docks_available) AS docks,
            any_value(capacity) AS capacity,
            any_value(status) = 'OPEN' AS open
        FROM (
            SELECT *, TRY_CAST(regexp_extract(name, '^(\\d+)', 1) AS INTEGER) AS station
            FROM {_collected()}
            WHERE at_utc >= '{t0}' AND at_utc < '{t1}'
        )
        WHERE station IS NOT NULL
        GROUP BY 1, 2
    """)

    con.execute(f"""
        CREATE TABLE stations AS
        SELECT
            TRY_CAST(regexp_extract(name, '^(\\d+)', 1) AS INTEGER) AS station,
            arg_max(trim(regexp_replace(name, '^\\d+\\s*-\\s*', '')), at_utc) AS name,
            arg_max(latitude, at_utc) AS lat,
            arg_max(longitude, at_utc) AS lon,
            arg_max(capacity, at_utc) AS capacity
        FROM {_collected()}
        WHERE station IS NOT NULL
          AND at_utc >= '{t0}'
          AND at_utc < '{t1}'
        GROUP BY 1
    """)
    # bornes atelier (coordonnées bidon) + stations qui n'ont jamais eu un seul vélo
    con.execute("""
        DELETE FROM stations
        WHERE name LIKE '%ATELIER%'
           OR station IN (SELECT station FROM snap GROUP BY 1 HAVING max(bikes) = 0)
    """)
    con.execute("DELETE FROM snap WHERE station NOT IN (SELECT station FROM stations)")
    con.execute(f"COPY (SELECT * FROM stations ORDER BY station) TO '{STATIONS_FILE}' (FORMAT parquet)")

    # grille seulement sur la durée de vie de chaque station, sinon les nouvelles stations
    # ont des mois de valeurs manquantes.
    # vraies tables et pas des CTE : sinon duckdb écrit des Go sur le disque pendant l'ASOF join
    con.execute("CREATE TABLE life AS SELECT station, min(ts) AS first, max(ts) AS last FROM snap GROUP BY 1")
    con.execute(f"""
        CREATE TABLE times AS
        SELECT unnest(generate_series(
            time_bucket(INTERVAL '15min', min(ts) - INTERVAL '{LATE}') + INTERVAL '15min', max(ts),
            INTERVAL '15min')) AS ts
        FROM snap
    """)
    con.execute(f"""
        CREATE TABLE grid AS
        SELECT l.station, t.ts FROM life l JOIN times t ON t.ts BETWEEN l.first - INTERVAL '{LATE}' AND l.last
    """)
    # relevé de plus de 30 min -> manquant, on recopie pas
    con.execute(f"""
        COPY (
            SELECT
                g.station, g.ts,
                CASE WHEN g.ts - s.ts <= INTERVAL '{tolerance}' THEN s.bikes END::SMALLINT AS bikes,
                CASE WHEN g.ts - s.ts <= INTERVAL '{tolerance}' THEN s.docks END::SMALLINT AS docks,
                s.capacity::SMALLINT AS capacity,
                CASE WHEN g.ts - s.ts <= INTERVAL '{tolerance}' THEN s.open END AS open
            FROM grid g ASOF LEFT JOIN snap s ON g.station = s.station AND g.ts + INTERVAL '{LATE}' >= s.ts
            ORDER BY g.station, g.ts
        ) TO '{GRID_FILE}' (FORMAT parquet)
    """)


def load_grid(stations=None, columns=None):
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    where = f"WHERE station IN ({', '.join(map(str, stations))})" if stations else ""
    cols = ", ".join(columns) if columns else "*"
    df = con.execute(f"SELECT {cols} FROM '{GRID_FILE}' {where}").df()
    df["ts"] = df["ts"].dt.tz_convert(TZ)
    for c in ("bikes", "docks", "capacity"):
        if c in df:
            df[c] = df[c].astype("float32")
    # station fermée = manquant
    if "open" in df:
        closed = df["open"].eq(False)
        df.loc[closed, [c for c in ("bikes", "docks") if c in df]] = float("nan")
        df = df.drop(columns="open")
    return df


def load_stations():
    return pd.read_parquet(STATIONS_FILE)


if __name__ == "__main__":
    build()
    grid = load_grid()
    print(f"{len(grid):,} rows, {grid['station'].nunique()} stations, "
          f"{grid['bikes'].isna().mean():.1%} missing")
