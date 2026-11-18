import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
from astropy.table import Table

import calistar

calistar_module = importlib.import_module("calistar.calistar")


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return b'{"info": {"version": "0.0.0"}}'


class FakeSimbad:
    def __init__(self):
        self.server = "simbad.cds.unistra.fr"

    def add_votable_fields(self, *fields):
        return None

    def query_object(self, query):
        return None


class FakeVizier:
    def __init__(self, *args, **kwargs):
        self.VIZIER_SERVER = "vizier.cds.unistra.fr"

    def query_object(self, query, **kwargs):
        return {}


class FakeJob:
    def __init__(self, results):
        self.results = results

    def get_results(self):
        return self.results


class FakeGaia:
    MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    @classmethod
    def launch_job(cls, *args, **kwargs):
        return FakeJob(Table())


@pytest.fixture
def patched_constructor(monkeypatch, tmp_path):
    monkeypatch.setattr(calistar_module, "Simbad", FakeSimbad)
    monkeypatch.setattr(calistar_module, "Vizier", FakeVizier)
    monkeypatch.setattr(calistar_module, "Gaia", FakeGaia)
    monkeypatch.setattr(
        calistar_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )
    monkeypatch.setattr(
        calistar_module.pooch,
        "retrieve",
        lambda **kwargs: str(Path(kwargs["path"]) / kwargs["fname"]),
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    return calistar.CaliStar(gaia_source="1")


def test_constructor_configures_api_without_network(patched_constructor):
    assert patched_constructor.gaia_source == 1
    assert patched_constructor.gaia_release == "DR3"
    assert patched_constructor.gaia_idx == 3
    assert patched_constructor.simbad.server == "simbad.cds.unistra.fr"


def test_constructor_rejects_unknown_gaia_release(monkeypatch, tmp_path):
    monkeypatch.setattr(calistar_module, "Simbad", FakeSimbad)
    monkeypatch.setattr(calistar_module, "Vizier", FakeVizier)
    monkeypatch.setattr(
        calistar_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )
    monkeypatch.setattr(
        calistar_module.pooch,
        "retrieve",
        lambda **kwargs: str(Path(kwargs["path"]) / kwargs["fname"]),
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(ValueError, match="DR2.*EDR3.*DR3"):
        calistar.CaliStar(gaia_source=1, gaia_release="DR1")


def test_target_star_uses_mocked_catalog_clients(patched_constructor, monkeypatch):
    gaia_result = Table(
        {
            "source_id": [1],
            "ref_epoch": [2016.0],
            "ra": [10.0],
            "ra_error": [1.0],
            "dec": [20.0],
            "dec_error": [1.0],
            "pmra": [2.0],
            "pmra_error": [0.1],
            "pmdec": [3.0],
            "pmdec_error": [0.1],
            "parallax": [4.0],
            "parallax_error": [0.2],
            "phot_g_mean_flux_over_error": [100.0],
            "phot_bp_mean_flux_over_error": [100.0],
            "phot_rp_mean_flux_over_error": [100.0],
            "phot_g_mean_mag": [10.0],
            "phot_bp_mean_mag": [10.5],
            "phot_rp_mean_mag": [9.5],
            "astrometric_excess_noise": [0.0],
        }
    )
    monkeypatch.setattr(
        calistar_module.Gaia,
        "launch_job",
        lambda *args, **kwargs: FakeJob(gaia_result),
    )
    monkeypatch.setattr(
        calistar_module.Table,
        "read",
        lambda *args, **kwargs: Table(),
    )

    target = patched_constructor.target_star(
        write_json=False, get_gaiaxp=False, allwise_catalog=True
    )

    assert target["Gaia ID"] == 1
    assert target["Gaia release"] == "DR3"
    assert target["GAIA/GAIA3.G"][0] == pytest.approx(10.0)


def test_find_calib_uses_mocked_gaia_and_catalog_clients(
    patched_constructor, monkeypatch, tmp_path
):
    source_id = patched_constructor.gaia_source
    (tmp_path / f"target_dr3_{source_id}.json").write_text(
        json.dumps(
            {
                "Gaia RA": [10.0],
                "Gaia Dec": [20.0],
                "GAIA/GAIA3.G": [10.0, 0.1],
            }
        ),
        encoding="utf-8",
    )

    gaia_result = Table(
        {
            "source_id": [2],
            "ra": [10.01],
            "dec": [20.01],
            "phot_g_mean_mag": [10.2],
            "astrometric_excess_noise": [0.0],
        }
    )
    monkeypatch.setattr(
        calistar_module.Gaia,
        "launch_job",
        lambda *args, **kwargs: FakeJob(gaia_result),
    )

    class FindVizier(FakeVizier):
        def query_object(self, query, **kwargs):
            return {
                "II/246/out": Table(
                    {
                        "_r": [0.1],
                        "Jmag": [10.0],
                        "Hmag": [10.0],
                        "Kmag": [10.0],
                    }
                ),
                "II/328/allwise": Table(
                    {
                        "_r": [0.1],
                        "W1mag": [10.0],
                        "W2mag": [10.0],
                        "W3mag": [10.0],
                        "W4mag": [10.0],
                    }
                ),
            }

    monkeypatch.setattr(calistar_module, "Vizier", FindVizier)

    result = patched_constructor.find_calib(
        search_radius=1.0, g_mag_range=(-1.0, 1.0), write_csv=False
    )

    assert result["Gaia ID"].tolist() == [2]
    assert result["2MASS/2MASS.H"].tolist() == [10.0]


def test_select_calib_filters_sources(patched_constructor, tmp_path):
    source_id = patched_constructor.gaia_source
    (tmp_path / f"target_dr3_{source_id}.json").write_text(
        json.dumps({"2MASS/2MASS.J": [10.0]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Gaia ID": [2, 3],
            "2MASS/2MASS.J": [10.05, 10.3],
        }
    ).to_csv(tmp_path / f"calib_find_dr3_{source_id}.csv", index=False)

    result = patched_constructor.select_calib(
        filter_names=["2MASS/2MASS.J"], mag_diff=0.1, write_csv=False
    )

    assert result["Gaia ID"].tolist() == [2]
