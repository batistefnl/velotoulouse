import pandas as pd

from velo import live


class Response:
    def __init__(self, hourly):
        self.hourly = hourly

    def raise_for_status(self):
        pass

    def json(self):
        return {"hourly": self.hourly}


def test_weather_falls_back_to_the_last_published_hour(monkeypatch):
    # à 10:20 UTC l'heure 9h-10h est pas encore publiée
    hourly = {"time": ["2024-11-12T08:00", "2024-11-12T09:00", "2024-11-12T10:00", "2024-11-12T11:00"],
              "precipitation": [0.0, 1.2, None, None], "temperature_2m": [9.0, 10.0, None, None]}
    monkeypatch.setattr(live.requests, "get", lambda *a, **k: Response(hourly))
    w = live.fetch_weather(pd.Timestamp("2024-11-12 10:20", tz="UTC"))
    assert w["rain"].tolist() == [1.2] and w["temp"].tolist() == [10.0]
    assert w.index[0] == pd.Timestamp("2024-11-12 08:00", tz="UTC")
