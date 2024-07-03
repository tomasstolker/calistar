import os
import pytest
import calistar


class TestCaliStar:
    def setup_class(self) -> None:

        self.limit = 1e-8
        self.test_dir = os.path.dirname(__file__) + "/"
        self.cal_star = calistar.CaliStar(gaia_source=6843672087120107264)

    def teardown_class(self):
        os.remove("calib_find_dr3_6843672087120107264.csv")
        os.remove("calib_select_dr3_6843672087120107264.csv")
        os.remove("gaiaxp_6843672087120107264_spec.dat")
        os.remove("gaiaxp_6843672087120107264_cov.dat")
        os.remove("gaiaxp_6843672087120107264.jpg")
        os.remove("target_dr3_6843672087120107264.json")

    def test_calistar(self) -> None:

        assert isinstance(self.cal_star, calistar.calistar.CaliStar)

    def test_target_star(self) -> None:

        self.cal_star.target_star(
            write_json=True, get_gaiaxp=True, allwise_catalog=True
        )

    def test_find_calib(self) -> None:

        self.cal_star.find_calib(search_radius=1.0, g_mag_range=None, write_csv=True)

    def test_select_calib(self) -> None:

        self.cal_star.select_calib(
            filter_names=["2MASS/2MASS.H"], mag_diff=5.0, write_csv=True
        )
