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
        "vizier_timeout": float(os.getenv("CALISTAR_TEST_VIZIER_TIMEOUT", "60.0")),
    }


@pytest.fixture
def cal_star(cds_servers):
    return calistar.CaliStar(
        gaia_source=6843672087120107264,
        **cds_servers,
    )


# separation warnings are OK for purpose of this test and only add noise
@pytest.mark.filterwarnings("ignore:The separation")
def test_calistar_workflow(cal_star) -> None:
    cal_star.target_star(write_json=True, get_gaiaxp=False, allwise_catalog=True)
    cal_star.find_calib(search_radius=1.0, g_mag_range=(-1.0, 1.0), write_csv=True)
    cal_star.select_calib(
        filter_names=["2MASS/2MASS.Ks", "WISE/WISE.W1"],
        mag_diff={"2MASS/2MASS.Ks": 0.3, "WISE/WISE.W1": 0.5},
    )
