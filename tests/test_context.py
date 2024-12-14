import pytest
import requests

from velo import context


def not_found():
    response = requests.Response()
    response.status_code = 404
    return requests.HTTPError(response=response)


def test_weather_file_falls_back_to_the_next_name(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "RAW", tmp_path)
    tried = []

    def get(url, path, refresh=False):
        tried.append(path.name)
        if path.name != context.METEO_FILES[-1]:
            raise not_found()
        return path

    monkeypatch.setattr(context, "_get", get)
    assert context._meteo_file().name == context.METEO_FILES[-1]
    assert tried == context.METEO_FILES


def test_local_weather_file_is_used_first(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "RAW", tmp_path)
    (tmp_path / context.METEO_FILES[1]).touch()
    monkeypatch.setattr(context, "_get", lambda *a, **k: pytest.fail("no download when a file is there"))
    assert context._meteo_file().name == context.METEO_FILES[1]
