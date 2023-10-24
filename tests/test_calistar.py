import os
import pytest
import calistar


class TestCaliStar:
    def setup_class(self) -> None:

        self.limit = 1e-8
        self.test_dir = os.path.dirname(__file__) + "/"
        self.cal_star = calistar.CaliStar(gaia_source=6843672087120107264)

    def test_calistar(self) -> None:

        assert isinstance(self.cal_star, calistar.calistar.CaliStar)

    def test_target_star(self) -> None:

        self.cal_star.target_star()

    def test_find_calib(self) -> None:

        self.cal_star.find_calib(search_radius=1.)

    def test_select_calib(self) -> None:

        self.cal_star.select_calib(filter_names=["2MASS H"], mag_diff=5.)
