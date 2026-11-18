import os

import pytest

import calistar

pytestmark = pytest.mark.integration


GAIA_SOURCE = 6199395656645840384


def teardown_module():
    for filename in (
        f"target_dr3_{GAIA_SOURCE}.json",
        f"calib_find_dr3_{GAIA_SOURCE}.csv",
        f"calib_select_dr3_{GAIA_SOURCE}.csv",
    ):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass


@pytest.fixture
def cds_servers():
    return {
        "simbad_server": os.getenv(
            "CALISTAR_TEST_SIMBAD_SERVER", "simbad.cds.unistra.fr"
        ),
        "vizier_server": os.getenv(
            "CALISTAR_TEST_VIZIER_SERVER", "vizier.cds.unistra.fr"
        ),
        "vizier_timeout": float(
            os.getenv("CALISTAR_TEST_VIZIER_TIMEOUT", "60.0")
        ),
    }


@pytest.fixture
def mirror_cal_star(cds_servers):
    return calistar.CaliStar(
        gaia_source=6199395656645840384,
        **cds_servers,
    )

def test_calistar_workflow(mirror_cal_star) -> None:
    mirror_cal_star.target_star(
        write_json=True, get_gaiaxp=False, allwise_catalog=True
    )
    mirror_cal_star.find_calib(
        search_radius=1.0, g_mag_range=(-1.0, 1.0), write_csv=True
    )
    mirror_cal_star.select_calib(
        filter_names=["2MASS/2MASS.H"], mag_diff=5.0, write_csv=True
    )
