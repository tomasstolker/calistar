import os
import pytest
import calistar


class TestCaliStar:
    def setup_class(self) -> None:

        self.gaia_source = 6199395656645840384
        self.limit = 1e-8
        self.test_dir = os.path.dirname(__file__) + "/"
        self.cal_star = calistar.CaliStar(gaia_source=self.gaia_source)

    def teardown_class(self):
        os.remove(f"target_dr3_{self.gaia_source}.json")
        os.remove(f"calib_find_dr3_{self.gaia_source}.csv")
        os.remove(f"calib_select_dr3_{self.gaia_source}.csv")
        # os.remove(f"gaiaxp_{self.gaia_source}_spec.dat")
        # os.remove(f"gaiaxp_{self.gaia_source}_cov.dat")
        # os.remove(f"gaiaxp_{self.gaia_source}.jpg")

    def test_calistar(self) -> None:

        assert isinstance(self.cal_star, calistar.calistar.CaliStar)

    def test_target_star(self) -> None:

        self.cal_star.target_star(
            write_json=True, get_gaiaxp=False, allwise_catalog=True
        )

    def test_find_calib(self) -> None:

        self.cal_star.find_calib(search_radius=1.0, g_mag_range=(-1.0, 1.0), write_csv=True)

    def test_select_calib(self) -> None:

        self.cal_star.select_calib(
            filter_names=["2MASS/2MASS.H"], mag_diff=5.0, write_csv=True
        )
