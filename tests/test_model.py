import numpy as np
import pandas as pd

from velo import features as F
from velo import model
from velo.config import TEST_START


def test_training_targets_stop_before_the_test_period():
    ts = pd.date_range(TEST_START - pd.Timedelta("2h"), TEST_START + pd.Timedelta("1h"), freq="15min")
    st = pd.DataFrame({"station": 1, "ts": ts, "bikes": np.float32(5), "docks": 5.0, "capacity": 10.0})
    df = F.station_features(F.add_calendar(st, pd.DataFrame(columns=["rain", "temp"])))
    train, test, _ = model.split(df)
    for h in F.HORIZONS:
        known = train.loc[train[f"y{h}"].notna(), "ts"]
        assert (known + pd.Timedelta(minutes=h) < TEST_START).all()
        assert known.max() + pd.Timedelta(minutes=h) == TEST_START - pd.Timedelta("15min")
