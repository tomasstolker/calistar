"""
Module with the ``calistar`` tool.
"""

import json
import socket
import urllib.request
import warnings

from typing import Dict, List, Optional, Tuple, Union

import astropy.units as u

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from rich import print as rprint
from rich.progress import track
from typeguard import typechecked

import calistar


Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"
Gaia.ROW_LIMIT = -1
Vizier.ROW_LIMIT = 1


class CaliStar:
    """
    Class for finding calibration stars based on their separation
    and magnitude difference with the requested ``gaia_source``.
    """

    @typechecked
    def __init__(
        self,
        gaia_source: Union[int, str],
    ) -> None:
        """
        Parameters
        ----------
        gaia_source : int, str
            The GAIA DR3 source ID of the object for which
            calibration source should be searched for.

        Returns
        -------
        NoneType
            None
        """

        calistar_init = f"[bold magenta]calistar v{calistar.__version__}[/bold magenta]"
        len_text = len(f"calistar v{calistar.__version__}")

        print(len_text * "=")
        rprint(calistar_init)
        print(len_text * "=")

        # Set attributes of CaliStar

        self.gaia_source = gaia_source  # Gaia DR3 source ID

        if isinstance(gaia_source, str):
            self.gaia_source = int(self.gaia_source)

        self.gaia_filters = ["GAIA G", "GAIA BP", "GAIA RP"]
        self.twomass_filters = ["2MASS J", "2MASS H", "2MASS Ks"]
        self.wise_filters = ["WISE W1", "WISE W2", "WISE W3", "WISE W4"]

        # Check if there is a new version available

        try:
            pypi_url = "https://pypi.org/pypi/calistar/json"

            with urllib.request.urlopen(pypi_url, timeout=1.0) as open_url:
                url_content = open_url.read()
                url_data = json.loads(url_content)
                latest_version = url_data["info"]["version"]

        except (urllib.error.URLError, socket.timeout):
            latest_version = None

        if latest_version is not None and calistar.__version__ != latest_version:
            print(f"\nA new version ({latest_version}) is available!")
            print("Want to stay informed about updates?")
            print("Please have a look at the Github page:")
            print("https://github.com/tomasstolker/calistar")

    @typechecked
    def target_star(
        self,
    ) -> Dict[str, Union[str, float]]:
        """
        Function for finding calibration stars.

        Parameters
        ----------

        Returns
        -------
        dict
            Dictionary with the target properties.
        """

        target_dict = {}

        # for table_item in Gaia.load_tables(only_names=True):
        #     print (table_item.get_qualified_name())

        # Gaia query for selected Gaia DR3 source

        gaia_query = f"""
        SELECT *
        FROM gaiadr3.gaia_source
        WHERE source_id = {self.gaia_source}
        """

        # Launch the Gaia job and get the results

        print("\n-> Querying GAIA DR3...\n")

        gaia_job = Gaia.launch_job_async(gaia_query, dump_to_file=False, verbose=False)
        gaia_result = gaia_job.get_results()

        target_dict["Gaia ID"] = int(gaia_result["source_id"][0])
        target_dict["Gaia RA"] = float(gaia_result["ra"][0])
        target_dict["Gaia Dec"] = float(gaia_result["dec"][0])
        target_dict["Gaia G mag"] = float(gaia_result["phot_g_mean_mag"][0])

        # Create SkyCoord object from the RA and Dec of the selected Gaia DR3 source

        gaia_coord = SkyCoord(
            gaia_result["ra"][0],
            gaia_result["dec"][0],
            frame="icrs",
            unit=(u.deg, u.deg),
        )

        coord_str = gaia_coord.to_string(
            "hmsdms", alwayssign=True, precision=2, pad=True
        )

        if self.gaia_source != gaia_result["source_id"][0]:
            raise ValueError(
                f"The requested source ID ({self.gaia_source}) "
                "is not equal to the retrieved source ID "
                f"({gaia_result['source_id'][0]})."
            )

        print(f"\nGAIA DR3 source ID = {gaia_result['source_id'][0]}")
        print(f"Reference epoch = {gaia_result['ref_epoch'][0]}")

        print(
            f"Parallax = {gaia_result['parallax'][0]:.2f} "
            f"+/- {gaia_result['parallax_error'][0]:.2f} mas"
        )

        print(f"\nRA = {gaia_result['ra'][0]:.6f} deg")
        print(f"Dec = {gaia_result['dec'][0]:.6f} deg")
        print(f"Coordinates = {coord_str}")

        print(
            f"\nProper motion RA = {gaia_result['pmra'][0]:.2f} "
            f"+/- {gaia_result['pmra_error'][0]:.2f} mas/yr"
        )
        print(
            f"Proper motion Dec = {gaia_result['pmdec'][0]:.2f} "
            f"+/- {gaia_result['pmdec_error'][0]:.2f} mas/yr"
        )
        print(
            f"Radial velocity = {gaia_result['radial_velocity'][0]:.2f} "
            f"+/- {gaia_result['radial_velocity_error'][0]:.2f} km/s"
        )

        print(f"\nG mean mag = {gaia_result['phot_g_mean_mag'][0]:.6f}")
        print(f"BP mean mag = {gaia_result['phot_bp_mean_mag'][0]:.6f}")
        print(f"RP mean mag = {gaia_result['phot_rp_mean_mag'][0]:.6f}")

        if not np.ma.is_masked(gaia_result["teff_gspphot"]):
            print(f"\nEffective temperature = {gaia_result['teff_gspphot'][0]:.0f} K")
            print(f"Surface gravity = {gaia_result['logg_gspphot'][0]:.2f}")
            print(f"Metallicity = {gaia_result['mh_gspphot'][0]:.2f}")
            print(f"G-band extinction = {gaia_result['ag_gspphot'][0]:.2f}")

        if gaia_result["non_single_star"][0] == 0:
            print("\nNon single star = False")

        elif gaia_result["non_single_star"][0] == 1:
            print("\nNon single star = True")

        else:
            warnings.warn(
                f"The 'non_single_star' value is {gaia_result['non_single_star'][0]}"
            )

        print(
            "Single star probability from DSC-Combmod = "
            f"{gaia_result['classprob_dsc_combmod_star'][0]:.2f}"
        )

        print(
            f"Astrometric excess noise = {gaia_result['astrometric_excess_noise'][0]:.2f}"
        )

        print(f"\nXP continuous = {gaia_result['has_xp_continuous'][0]}")
        print(f"XP sampled = {gaia_result['has_xp_sampled'][0]}")
        print(f"RVS spectrum = {gaia_result['has_rvs'][0]}")

        # Add spectral type and 2MASS JHKs magnitudes to the Simbad output

        # print(Simbad.list_votable_fields())

        Simbad.add_votable_fields(
            "sptype",
            "flux(J)",
            "flux(H)",
            "flux(K)",
            "flux_error(J)",
            "flux_error(H)",
            "flux_error(K)",
        )

        # Simbad query for selected Gaia DR3 source

        print("\n-> Querying Simbad...\n")
        simbad_result = Simbad.query_object(f"GAIA DR3 {self.gaia_source}")
        simbad_result = simbad_result[0]

        # print(simbad_result.columns)

        print(f"Simbad ID = {simbad_result['MAIN_ID']}")

        print(f"Spectral type = {simbad_result['SP_TYPE']}")

        print(
            f"\n2MASS J mag = {simbad_result['FLUX_J']:.3f} "
            f"+/- {simbad_result['FLUX_ERROR_J']:.3f}"
        )

        print(
            f"2MASS H mag = {simbad_result['FLUX_H']:.3f} "
            f"+/- {simbad_result['FLUX_ERROR_H']:.3f}"
        )

        print(
            f"2MASS Ks mag = {simbad_result['FLUX_K']:.3f} "
            f"+/- {simbad_result['FLUX_ERROR_K']:.3f}"
        )

        target_dict["Simbad ID"] = simbad_result["MAIN_ID"]
        target_dict["SpT"] = simbad_result["SP_TYPE"]
        target_dict["2MASS J"] = float(simbad_result["FLUX_J"])
        target_dict["2MASS H"] = float(simbad_result["FLUX_H"])
        target_dict["2MASS Ks"] = float(simbad_result["FLUX_K"])

        # VizieR query for selected Gaia DR3 source
        # Sort the result by distance from the queried object

        print("\n-> Querying VizieR...\n")

        vizier_obj = Vizier(columns=["*", "+_r"], catalog="II/328/allwise")

        radius = u.Quantity(1.0 * u.arcmin)

        vizier_result = vizier_obj.query_object(
            f"GAIA DR3 {self.gaia_source}", radius=radius
        )
        vizier_result = vizier_result["II/328/allwise"]

        # print(f"Found {len(vizier_result)} object(s) in VizieR. Selecting the first object...")
        vizier_result = vizier_result[0]

        print(f"ALLWISE source ID = {vizier_result['AllWISE']}")

        print(
            f"Separation between Gaia and ALLWISE source = "
            f"{1e3*vizier_result['_r']:.1f} mas"
        )

        print(
            f"\nWISE W1 mag = {vizier_result['W1mag']:.3f} "
            f"+/- {vizier_result['e_W1mag']:.3f}"
        )

        print(
            f"WISE W2 mag = {vizier_result['W2mag']:.3f} "
            f"+/- {vizier_result['e_W2mag']:.3f}"
        )

        print(
            f"WISE W3 mag = {vizier_result['W3mag']:.3f} "
            f"+/- {vizier_result['e_W3mag']:.3f}"
        )

        print(
            f"WISE W4 mag = {vizier_result['W4mag']:.3f} "
            f"+/- {vizier_result['e_W4mag']:.3f}"
        )

        target_dict["WISE W1"] = float(vizier_result["W1mag"])
        target_dict["WISE W2"] = float(vizier_result["W2mag"])
        target_dict["WISE W3"] = float(vizier_result["W3mag"])
        target_dict["WISE W4"] = float(vizier_result["W4mag"])

        json_file = f"target_{self.gaia_source}.json"

        with open(json_file, "w", encoding="utf-8") as open_file:
            json.dump(target_dict, open_file, indent=4)

        return target_dict

    @typechecked
    def find_calib(
        self,
        search_radius: float = 0.1,
        mag_range: Optional[Tuple[float, float]] = None,
    ) -> pd.DataFrame:
        """
        Function for finding calibration stars.

        Parameters
        ----------
        search_radius : float
            Radius (in degrees) of the cone that is used to
            query the GAIA DR3 catalog to search for
            calibration source in the vicinity of the
            selected ``gaia_source`` (default: 0.1).
        mag_range : tuple(float, float), None
            Magnitude range in the Gaia G band that is used
            for querying sources in the Gaia DR3 catalog.
            A range of +/- 1 mag relative to the G-band
            magnitude of the ``gaia_source`` is used if
            the argument of ``mag_range`` is set to ``None``.

        Returns
        -------
        pandas.DataFrame
            The ``DataFrame`` with the list of queried sources.
        """

        json_file = f"target_{self.gaia_source}.json"

        with open(json_file, "r", encoding="utf-8") as open_file:
            target_dict = json.load(open_file)

        if mag_range is None:
            mag_range = (
                target_dict["Gaia G mag"] - 1.0,
                target_dict["Gaia G mag"] + 1.0,
            )

        # Add 2MASS JHKs magnitudes to the Simbad output

        # print(Simbad.list_votable_fields())

        Simbad.add_votable_fields(
            "sptype",
            "flux(J)",
            "flux(H)",
            "flux(K)",
            "flux_error(J)",
            "flux_error(H)",
            "flux_error(K)",
        )

        print("\n-> Gaia cone search\n")

        # gaia_coord = SkyCoord(
        #     target_dict["Gaia RA"],
        #     target_dict["Gaia Dec"],
        #     frame="icrs",
        #     unit=(u.deg, u.deg),
        # )

        # radius = u.Quantity(search_radius * u.deg)
        # gaia_job = Gaia.cone_search_async(gaia_coord, radius=radius)

        print(f"\nRadius of search cone = {search_radius} deg")

        gaia_query = f"""
        SELECT *, DISTANCE({target_dict['Gaia RA']},
        {target_dict['Gaia Dec']}, ra, dec) AS ang_sep
        FROM gaiadr3.gaia_source
        WHERE DISTANCE({target_dict['Gaia RA']},
        {target_dict['Gaia Dec']}, ra, dec) < {search_radius}
        AND phot_g_mean_mag > {mag_range[0]}
        AND phot_g_mean_mag < {mag_range[1]}
        AND parallax IS NOT NULL
        ORDER BY ang_sep ASC
        """

        # Launch the Gaia job and get the results

        print("\n-> Querying GAIA DR3...\n")

        gaia_job = Gaia.launch_job_async(gaia_query, dump_to_file=False, verbose=False)
        gaia_results = gaia_job.get_results()
        print(f"Found {len(gaia_results)} object(s) with cone search")

        print("\n-> Finding calibration stars...\n")

        columns = ["Simbad ID", "Gaia ID", "SpT"]

        columns += self.gaia_filters
        columns += self.twomass_filters
        columns += self.wise_filters

        columns += [
            "Non single star",
            "Single star probability",
            "Astrometric excess noise",
        ]

        # Initiate all values in the dataframe to NaN
        cal_df = pd.DataFrame(index=range(len(gaia_results)), columns=columns)

        drop_indices = []

        warnings.filterwarnings("ignore", category=UserWarning)

        vizier_obj = Vizier(columns=["*", "+_r"], catalog="II/328/allwise")

        for result_item in track(gaia_results, description="Processing..."):
            cal_df.loc[result_item.index, "Gaia ID"] = result_item["source_id"]

            cal_df.loc[result_item.index, "GAIA G"] = result_item["phot_g_mean_mag"]

            cal_df.loc[result_item.index, "GAIA BP"] = result_item["phot_bp_mean_mag"]

            cal_df.loc[result_item.index, "GAIA RP"] = result_item["phot_rp_mean_mag"]

            simbad_result = Simbad.query_object(f"GAIA DR3 {result_item['source_id']}")

            if simbad_result is not None:
                simbad_result = simbad_result[0]

                if np.ma.is_masked(simbad_result["MAIN_ID"]):
                    cal_df.loc[result_item.index, "Simbad ID"] = np.nan
                else:
                    cal_df.loc[result_item.index, "Simbad ID"] = simbad_result["MAIN_ID"]

                if np.ma.is_masked(simbad_result["SP_TYPE"]):
                    cal_df.loc[result_item.index, "SpT"] = np.nan
                else:
                    cal_df.loc[result_item.index, "SpT"] = simbad_result["SP_TYPE"]

                if np.ma.is_masked(simbad_result["FLUX_J"]):
                    cal_df.loc[result_item.index, "2MASS J"] = np.nan
                else:
                    cal_df.loc[result_item.index, "2MASS J"] = simbad_result["FLUX_J"]

                if np.ma.is_masked(simbad_result["FLUX_H"]):
                    cal_df.loc[result_item.index, "2MASS H"] = np.nan
                else:
                    cal_df.loc[result_item.index, "2MASS H"] = simbad_result["FLUX_H"]

                if np.ma.is_masked(simbad_result["FLUX_K"]):
                    cal_df.loc[result_item.index, "2MASS Ks"] = np.nan
                else:
                    cal_df.loc[result_item.index, "2MASS Ks"] = simbad_result["FLUX_K"]

            radius = u.Quantity(1.0 * u.arcmin)

            vizier_result = vizier_obj.query_object(
                f"GAIA DR3 {result_item['source_id']}",
                radius=radius,
                catalog="II/328/allwise",
            )

            if len(vizier_result) == 1:
                vizier_result = vizier_result["II/328/allwise"][0]

                # Separation between Gaia and ALLWISE source is more than 100 mas
                if vizier_result["_r"] > 0.1:
                    skip_source = True
                else:
                    skip_source = False

                if skip_source or np.ma.is_masked(vizier_result["W1mag"]):
                    cal_df.loc[result_item.index, "WISE W1"] = np.nan
                else:
                    cal_df.loc[result_item.index, "WISE W1"] = vizier_result["W1mag"]

                if skip_source or np.ma.is_masked(vizier_result["W2mag"]):
                    cal_df.loc[result_item.index, "WISE W2"] = np.nan
                else:
                    cal_df.loc[result_item.index, "WISE W2"] = vizier_result["W2mag"]

                if skip_source or np.ma.is_masked(vizier_result["W3mag"]):
                    cal_df.loc[result_item.index, "WISE W3"] = np.nan
                else:
                    cal_df.loc[result_item.index, "WISE W3"] = vizier_result["W3mag"]

                if skip_source or np.ma.is_masked(vizier_result["W4mag"]):
                    cal_df.loc[result_item.index, "WISE W4"] = np.nan
                else:
                    cal_df.loc[result_item.index, "WISE W4"] = vizier_result["W4mag"]

            else:
                drop_indices.append(result_item.index)

            cal_df.loc[result_item.index, "Non single star"] = result_item[
                "non_single_star"
            ]
            cal_df.loc[result_item.index, "Single star probability"] = result_item[
                "classprob_dsc_combmod_star"
            ]
            cal_df.loc[result_item.index, "Astrometric excess noise"] = result_item[
                "astrometric_excess_noise"
            ]

        warnings.filterwarnings("default", category=UserWarning)

        cal_df = cal_df.drop(index=drop_indices)
        cal_df["Gaia ID"] = cal_df["Gaia ID"].astype("int")

        output_file = f"calib_find_{self.gaia_source}.csv"

        print(f"\nStoring output: {output_file}")

        cal_df.to_csv(path_or_buf=output_file, header=True, index=False)

        return cal_df

    @typechecked
    def select_calib(
        self,
        filter_names: Optional[List[str]] = None,
        mag_diff: Union[float, Dict[str, float]] = 0.1,
    ) -> pd.DataFrame:
        """
        Function for selecting the calibration stars.

        Parameters
        ----------
        filter_names : list(str), None
            List with the filter names (default: ['2MASS J',
            '2MASS H', '2MASS Ks']).
        mag_diff : float
            Allowed magnitude difference between the selected target
            as argument of ``gaia_source`` and the queried sources
            with :func:`~calistar.calistar.CaliStar.find_calib()`
            (default: 0.1).

        Returns
        -------
        pandas.DataFrame
            The ``DataFrame`` with the selected calibration stars.
        """

        json_file = f"target_{self.gaia_source}.json"

        with open(json_file, "r", encoding="utf-8") as open_file:
            target_dict = json.load(open_file)

        if filter_names is None:
            filter_names = ["2MASS J", "2MASS H", "2MASS Ks"]

        cal_df = pd.read_csv(
            filepath_or_buffer=f"calib_find_{self.gaia_source}.csv",
            header=0,
            index_col=False,
        )

        print("\n-> Selecting calibration stars...\n")

        drop_indices = []

        for row_idx in track(range(len(cal_df)), description="Processing..."):
            for filter_item in filter_names:
                if np.isnan(cal_df.loc[row_idx, filter_item]) or (
                    np.abs(cal_df.loc[row_idx, filter_item] - target_dict[filter_item])
                    > mag_diff
                ):
                    if row_idx not in drop_indices:
                        drop_indices.append(row_idx)

        cal_df = cal_df.drop(index=drop_indices)

        output_file = f"calib_select_{self.gaia_source}.csv"

        print(f"\nStoring output: {output_file}")

        cal_df.to_csv(path_or_buf=output_file, header=True, index=False)

        return cal_df
