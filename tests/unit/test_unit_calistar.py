import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
from astropy.table import Table

import calistar

calistar_module = importlib.import_module("calistar.calistar")

from fakes import FakeGaia, FakeJob, FakeResponse, FakeSimbad, FakeVizier


@pytest.fixture(autouse=True)
def patched_dependencies(monkeypatch, tmp_path):
    # Replace all classes and methods that require network with
    # dummy instances that return predefined values
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


def test_attributes():
    star = calistar.CaliStar(gaia_source="1")
    assert star.gaia_source == 1
    assert star.gaia_release == "DR3"
    assert star.gaia_idx == 3
    assert star.simbad.server == "simbad.cds.unistra.fr"


def test_constructor_rejects_unknown_gaia_release():
    with pytest.raises(ValueError, match="DR2.*EDR3.*DR3"):
        calistar.CaliStar(gaia_source=1, gaia_release="DR1")


@pytest.fixture
def gaia_result():
    """Gaia result to use in mock test"""
    return Table(
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


def test_target_star_gaia_results(gaia_result, monkeypatch):
    """Test that target_star returns the properly formatted gaia results"""
    # Patch Gaia to return our mock result
    monkeypatch.setattr(
        calistar_module.Gaia,
        "launch_job",
        lambda *args, **kwargs: FakeJob(gaia_result),
    )

    star = calistar.CaliStar(gaia_source="1")
    target = star.target_star(write_json=False, get_gaiaxp=False, allwise_catalog=True)

    assert target["Gaia ID"] == 1
    assert target["Gaia release"] == "DR3"
    assert target["GAIA/GAIA3.G"][0] == pytest.approx(10.0)


@pytest.fixture
def calib_star_data():
    """Fake gaia and vizier results for a calibration star"""
    gaia = Table(
        {
            "source_id": [2],
            "ra": [10.01],
            "dec": [20.01],
            "phot_g_mean_mag": [10.2],
            "astrometric_excess_noise": [0.0],
        }
    )
    vizier = {
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
    return {"gaia": gaia, "vizier": vizier}


def test_find_calib_gaia_vizier_results(calib_star_data, monkeypatch, tmp_path):
    """Test that find_calib processes Gaia and Vizier results correctly"""
    # Create target star for which we'll search a calibration star
    star = calistar.CaliStar(gaia_source="1")

    # Create the target json that star.target_star() would normally create
    target_json_path = tmp_path / f"target_dr3_{star.gaia_source}.json"
    target_json_path.write_text(
        json.dumps(
            {
                "Gaia RA": [10.0],
                "Gaia Dec": [20.0],
                "GAIA/GAIA3.G": [10.0, 0.1],
            }
        ),
        encoding="utf-8",
    )

    # Patch Vizier and Gaia results with mock data and no network
    gaia_result = calib_star_data["gaia"]
    vizier_result = calib_star_data["vizier"]
    monkeypatch.setattr(
        calistar_module.Gaia,
        "launch_job",
        lambda *args, **kwargs: FakeJob(gaia_result),
    )
    vizier = FakeVizier()
    monkeypatch.setattr(
        vizier,
        "query_object",
        lambda *args, **kwargs: vizier_result,
    )
    monkeypatch.setattr(calistar_module, "Vizier", lambda *args, **kwargs: vizier)

    result = star.find_calib(
        search_radius=1.0, g_mag_range=(-1.0, 1.0), write_csv=False
    )

    assert result["Gaia ID"].tolist() == gaia_result["source_id"]
    assert result["2MASS/2MASS.H"].tolist() == vizier_result["II/246/out"]["Hmag"]


def test_select_calib_filters_sources(tmp_path):
    """Test that select_calib correctly selects a calibration star"""
    star = calistar.CaliStar(gaia_source="1")
    target_json_path = tmp_path / f"target_dr3_{star.gaia_source}.json"
    target_json_path.write_text(
        json.dumps({"2MASS/2MASS.J": [10.0]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Gaia ID": [2, 3],
            "2MASS/2MASS.J": [10.05, 10.3],
        }
    ).to_csv(tmp_path / f"calib_find_dr3_{star.gaia_source}.csv", index=False)

    result = star.select_calib(
        filter_names=["2MASS/2MASS.J"], mag_diff=0.1, write_csv=False
    )

    assert result["Gaia ID"].tolist() == [2]
