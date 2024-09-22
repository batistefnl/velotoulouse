import pandas as pd
import pytest

from velo import context


@pytest.fixture(autouse=True)
def no_network_calendars(monkeypatch):
    # pas de réseau dans les tests
    tz = "Europe/Paris"
    monkeypatch.setattr(context, "public_holidays", lambda years=None: pd.DatetimeIndex(["2024-11-11"]).tz_localize(tz))
    monkeypatch.setattr(context, "school_holidays", lambda until=None: [
        (pd.Timestamp("2024-10-19", tz=tz), pd.Timestamp("2024-11-04", tz=tz))])
