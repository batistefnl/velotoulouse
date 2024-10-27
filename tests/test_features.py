import numpy as np
import pandas as pd

from velo import features as F

TZ = "Europe/Paris"


def station(bikes, start="2024-09-16 08:00"):
    ts = pd.date_range(start, periods=len(bikes), freq="15min", tz=TZ)
    return pd.DataFrame({"station": 1, "ts": ts, "bikes": np.array(bikes, dtype="float32"),
                         "docks": 10.0, "capacity": 12.0})


def test_targets_look_ahead_and_lags_look_back():
    f = F.station_features(station([0, 1, 2, 3, 4, 5]))
    assert f["y15"].tolist()[:5] == [False, True, True, True, True]
    assert pd.isna(f["y15"].iloc[-1])
    assert f["y60"].tolist()[:2] == [True, True]
    assert f["d15"].iloc[1] == 1 and pd.isna(f["d15"].iloc[0])
    assert f["d60"].iloc[4] == 4


def test_target_is_missing_when_future_is_missing():
    f = F.station_features(station([5, np.nan, 5]))
    assert pd.isna(f["y15"].iloc[0])


def test_weather_comes_from_the_previous_hour():
    weather = pd.DataFrame({"rain": [1.0, 7.0], "temp": [10.0, 11.0]},
                           index=pd.date_range("2024-09-16 07:00", periods=2, freq="h", tz=TZ))
    df = F.add_calendar(station([1, 1, 1, 1, 1]), weather)
    # 8:00-8:45 -> heure 7h-8h, 9:00 -> 8h-9h
    assert df["rain"].tolist() == [1.0, 1.0, 1.0, 1.0, 7.0]


def test_calendar_handles_the_ambiguous_dst_hour():
    df = F.add_calendar(station([1] * 12, start="2024-10-27 00:00"), pd.DataFrame(columns=["rain", "temp"]))
    assert df["school_holiday"].all()
    assert df["slot"].iloc[0] == 0 and df["slot"].iloc[-1] == 11


def test_profile_is_probability_per_station_day_type_and_slot():
    a = station([0, 5], start="2024-09-16 08:00")  # Monday
    b = station([5, 5], start="2024-09-17 08:00")  # Tuesday
    df = F.add_calendar(pd.concat([a, b], ignore_index=True), pd.DataFrame(columns=["rain", "temp"]))
    table = F.profile_table(df)
    assert table[(1, "work", 32)] == 0.5
    assert table[(1, "work", 33)] == 1.0
    assert F.add_profile(df, table)["profile"].tolist() == [0.5, 1.0, 0.5, 1.0]


def test_public_holiday_is_a_day_off():
    df = F.add_calendar(station([1], start="2024-11-11 08:00"), pd.DataFrame(columns=["rain", "temp"]))
    assert df["day_type"].iloc[0] == "off"
