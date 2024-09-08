import pandas as pd
import pytest

from velo import data


def snapshot(ts, name, bikes, lat=43.6, capacity=10, status="OPEN"):
    return {"name": name, "latitude": lat, "longitude": 1.44, "capacity": capacity,
            "num_bikes_available": bikes, "num_docks_available": capacity - bikes, "status": status,
            "collected_at_utc": pd.Timestamp(ts, tz="UTC")}


@pytest.fixture
def collected(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    monkeypatch.setattr(data, "RAW", raw)
    monkeypatch.setattr(data, "PROCESSED", processed)
    monkeypatch.setattr(data, "GRID_FILE", processed / "grid.parquet")
    monkeypatch.setattr(data, "STATIONS_FILE", processed / "stations.parquet")
    rows = [
        snapshot("2024-09-10 10:02", "00224 - CAMPUS SUPAERO", 5),
        snapshot("2024-09-10 10:40", "00224 - BELIN - SUPAERO", 3),
        snapshot("2024-09-10 11:20", "00224 - BELIN - SUPAERO", 0, status="CLOSED"),
        snapshot("2024-09-10 11:40", "00224 - BELIN - SUPAERO", 2),
        snapshot("2024-09-10 10:02", "01031 - BORNE CGB ATELIER", 4),
        snapshot("2024-09-10 11:20", "01031 - BORNE CGB ATELIER", 4),
        snapshot("2024-09-10 10:02", "NARBONNE - SAHUQUE", 2),
    ]
    pd.DataFrame(rows).to_parquet(raw / "collected_2024-09-10.parquet")
    start, end = pd.Timestamp("2024-09-01", tz="UTC"), pd.Timestamp("2024-10-01", tz="UTC")
    data.build(start, end)
    return data.load_grid().set_index("ts")


def test_station_identity_survives_renaming(collected):
    assert set(collected["station"]) == {224}
    assert data.load_stations().loc[0, "name"] == "BELIN - SUPAERO"


def t(hm):
    return pd.Timestamp(f"2024-09-10 {hm}", tz="UTC")


def test_grid_takes_last_snapshot_within_tolerance(collected):
    bikes = collected["bikes"]
    assert bikes[t("10:15")] == 5
    assert bikes[t("10:30")] == 5  # 28 minutes old, still used
    assert bikes[t("10:45")] == 3
    assert pd.isna(bikes[t("11:15")])  # 35 minutes old: missing, not carried forward


def test_snapshot_seconds_after_the_quarter_hour_is_its_state(collected):
    name = "00224 - BELIN - SUPAERO"
    rows = [snapshot("2024-09-11 00:00:02", name, 6), snapshot("2024-09-11 00:15:02", name, 7)]
    pd.DataFrame(rows).to_parquet(data.RAW / "collected_2024-09-11.parquet")
    data.build(pd.Timestamp("2024-09-01", tz="UTC"), pd.Timestamp("2024-10-01", tz="UTC"))
    bikes = data.load_grid().set_index("ts")["bikes"]
    assert bikes[pd.Timestamp("2024-09-11 00:00", tz="UTC")] == 6
    assert bikes[pd.Timestamp("2024-09-11 00:15", tz="UTC")] == 7


def test_closed_station_counts_as_missing_without_other_gaps(collected):
    # cas où il y a aucun manquant dans la sélection, le fermé doit quand même devenir NaN
    name = "00224 - BELIN - SUPAERO"
    rows = [snapshot(f"2024-09-10 10:{m}:02", name, 5, status="CLOSED" if m == "15" else "OPEN")
            for m in ("00", "15", "30")]
    pd.DataFrame(rows).to_parquet(data.RAW / "collected_2024-09-10.parquet")
    data.build(pd.Timestamp("2024-09-01", tz="UTC"), pd.Timestamp("2024-10-01", tz="UTC"))
    bikes = data.load_grid([224], columns=["station", "ts", "bikes", "open"]).set_index("ts")["bikes"]
    assert pd.isna(bikes[t("10:15")]) and bikes[t("10:30")] == 5


def test_closed_station_counts_as_missing(collected):
    # 11:30 prend le relevé de 11:20, station fermée
    assert pd.isna(collected["bikes"][t("11:30")])


def test_daily_files_are_read_together(collected):
    next_day = pd.DataFrame([snapshot(f"2024-09-11 00:{m}", "00224 - BELIN - SUPAERO", 7) for m in (10, 20)])
    next_day.to_parquet(data.RAW / "collected_2024-09-11.parquet")
    data.build(pd.Timestamp("2024-09-01", tz="UTC"), pd.Timestamp("2024-10-01", tz="UTC"))
    assert data.load_grid().set_index("ts")["bikes"][pd.Timestamp("2024-09-11 00:15", tz="UTC")] == 7


def test_missing_collected_files_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "RAW", tmp_path)
    with pytest.raises(FileNotFoundError):
        data.build()
