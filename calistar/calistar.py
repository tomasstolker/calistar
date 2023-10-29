"""
Module with the ``calistar`` tool.
"""

import json
import socket
import urllib.request
import warnings

from copy import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
import pandas as pd
import pooch

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from rich import print as rprint
from rich.progress import track
from typeguard import typechecked

import calistar


# No limit on the number of rows with a Gaia query
Gaia.ROW_LIMIT = -1

# Only return the nearest source with a Vizier query
Vizier.ROW_LIMIT = 1


class CaliStar:
    """
    Class for finding calibration stars based on their separation
    and magnitude difference with the selected ``gaia_source``.
    """

    @typechecked
    def __init__(
        self,
        gaia_source: Union[int, str],
        gaia_release: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        gaia_source : int, str
            The GAIA source ID of the object for which calibration
            sources should be searched for. For example, set the
            argument to ``6843672087120107264`` or
            ``"6843672087120107264"`` for the star HD 206893 when
            the DR2 or DR3 catalog as argument of ``gaia_release``.
        gaia_release : str, None
            Data release of the Gaia catalog that will be used for the
            queries (``"DR2"``, ``"EDR3"``, or ``"DR3"``). The default
            release is DR3 when the argument is set to ``None``.

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

        self.gaia_source = gaia_source  # Gaia source ID

        if isinstance(gaia_source, str):
            self.gaia_source = int(self.gaia_source)

        # Gaia data release version

        self.gaia_release = gaia_release

        if self.gaia_release is None:
            self.gaia_release = "DR3"

        if self.gaia_release not in ["DR2", "EDR3", "DR3"]:
            raise ValueError(
                "The argument of 'gaia_release' should "
                "be set to 'DR2', 'EDR3', or 'DR3'."
            )

        # Set the Gaia source catalog

        if self.gaia_release == "DR2":
            Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
            self.gaia_idx = 2

        elif self.gaia_release == "EDR3":
            Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"
            self.gaia_idx = 3

        elif self.gaia_release == "DR3":
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            self.gaia_idx = 3

        # Filter IDs from the SVO Filter Profile Service

        self.all_filters = [
            f"GAIA/GAIA{self.gaia_idx}.G",
            "2MASS/2MASS.J",
            "2MASS/2MASS.H",
            "2MASS/2MASS.Ks",
            "WISE/WISE.W1",
            "WISE/WISE.W2",
            "WISE/WISE.W3",
            "WISE/WISE.W4",
        ]

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

        # Create .calistar folder in the home directory

        self.calistar_folder = Path.home() / ".calistar"

        if not self.calistar_folder.exists():
            print(f"\nCreating .calistar folder in {Path.home()}...")
            self.calistar_folder.mkdir(parents=False, exist_ok=False)

        # Download The Washington Visual Double Star Catalog
        # https://cdsarc.cds.unistra.fr/viz-bin/cat/B/wds

        self.wds_file = Path.home() / ".calistar/wds_catalog.hdf5"

        url = "https://home.strw.leidenuniv.nl/~stolker/calistar/wds_catalog.hdf5"

        if not self.wds_file.exists():
            pooch.retrieve(
                url=url,
                known_hash="b0a63ed95bf060cccc3ce66afb36ed669d287ad85211cee9cde3bd57d48c622b",
                fname="wds_catalog.hdf5",
                path=self.calistar_folder,
                progressbar=True,
            )

    @typechecked
    def target_star(
        self,
        write_json: bool = True,
    ) -> Dict[str, Union[str, float]]:
        """
        Function for retrieving the the astrometric and
        photometric properties of a target star of interest. The
        function returns a dictionary with the properties, but it
        also (optionally) stores the data in a JSON file in the
        working folder.

        Parameters
        ----------
        write_json : bool
            Write the target properties to a JSON file (default: True).
            The file will be stored in the working folder and starts
            with ``target_``. The filename contains also the Gaia
            release and the Gaia source ID of the target.

        Returns
        -------
        dict
            Dictionary with the properties of the target star.
        """

        target_dict = {}

        # List all Gaia tables
        # for table_item in Gaia.load_tables(only_names=True):
        #     print (table_item.get_qualified_name())

        # Gaia query for selected Gaia source ID

        gaia_query = f"""
        SELECT *
        FROM gaia{self.gaia_release.lower()}.gaia_source
        WHERE source_id = {self.gaia_source}
        """

        # Launch the Gaia job and get the results

        print(f"\n-> Querying GAIA {self.gaia_release}...\n")

        gaia_job = Gaia.launch_job_async(gaia_query, dump_to_file=False, verbose=False)
        gaia_result = gaia_job.get_results()

        # print(gaia_result.columns)

        if self.gaia_release == "DR2":
            # Gaia DR2 VEGAMAG zero points
            # https://www.cosmos.esa.int/web/gaia/iow_20180316

            gaia_g_zp = (25.6914396869, 0.0011309370)
            gaia_bp_zp = (25.3488107670, 0.0004899854)
            gaia_rp_zp = (24.7626744847, 0.0035071711)

        elif self.gaia_release in ["EDR3", "DR3"]:
            # Gaia (E)DR3 VEGAMAG zero points
            # https://www.cosmos.esa.int/web/gaia/edr3-passbands

            gaia_g_zp = (25.6873668671, 0.0027553202)
            gaia_bp_zp = (25.3385422158, 0.0027901700)
            gaia_rp_zp = (24.7478955012, 0.0037793818)

        # Magnitude error, assuming Delta_f << f
        # Delta_m = -2.5/ln(10) Delta_f/f
        # Add in quadrature the uncertainty on the zero point

        mag_g_error = (
            -2.5 / np.log(10.0) / gaia_result["phot_g_mean_flux_over_error"][0]
        )

        mag_g_error = np.sqrt(mag_g_error**2 + gaia_g_zp[1] ** 2)

        mag_bp_error = (
            -2.5 / np.log(10.0) / gaia_result["phot_bp_mean_flux_over_error"][0]
        )

        mag_bp_error = np.sqrt(mag_bp_error**2 + gaia_bp_zp[1] ** 2)

        mag_rp_error = (
            -2.5 / np.log(10.0) / gaia_result["phot_rp_mean_flux_over_error"][0]
        )

        mag_rp_error = np.sqrt(mag_rp_error**2 + gaia_rp_zp[1] ** 2)

        target_dict["Gaia ID"] = int(gaia_result["source_id"][0])

        target_dict["Gaia release"] = self.gaia_release

        target_dict["Gaia epoch"] = gaia_result["ref_epoch"][0]

        target_dict["Gaia RA"] = (
            float(gaia_result["ra"][0]),  # (deg)
            float(gaia_result["dec_error"][0] / 3600.0),  # (deg)
        )

        target_dict["Gaia Dec"] = (
            float(gaia_result["dec"][0]),  # (deg)
            float(gaia_result["dec_error"][0] / 3600.0),  # (deg)
        )

        target_dict["Gaia pm RA"] = (
            float(gaia_result["pmra"][0]),  # (mas yr-1)
            float(gaia_result["pmra_error"][0]),  # (mas yr-1)
        )

        target_dict["Gaia pm Dec"] = (
            float(gaia_result["pmdec"][0]),  # (mas yr-1)
            float(gaia_result["pmdec_error"][0]),  # (mas yr-1)
        )

        target_dict["Gaia parallax"] = (
            float(gaia_result["parallax"][0]),  # (mas)
            float(gaia_result["parallax_error"][0]),  # (mas)
        )

        target_dict[f"GAIA/GAIA{self.gaia_idx}.G"] = (
            float(gaia_result["phot_g_mean_mag"][0]),
            mag_g_error,
        )

        target_dict[f"GAIA/GAIA{self.gaia_idx}.Gbp"] = (
            float(gaia_result["phot_bp_mean_mag"][0]),
            mag_bp_error,
        )

        target_dict[f"GAIA/GAIA{self.gaia_idx}.Grp"] = (
            float(gaia_result["phot_rp_mean_mag"][0]),
            mag_rp_error,
        )

        target_dict[f"GAIA/GAIA{self.gaia_idx}.Grp"] = (
            float(gaia_result["phot_rp_mean_mag"][0]),
            mag_rp_error,
        )

        if "grvs_mag" in gaia_result.columns:
            target_dict[f"GAIA/GAIA{self.gaia_idx}.Grvs"] = (
                float(gaia_result["grvs_mag"][0]),
                float(gaia_result["grvs_mag_error"][0]),
            )

        # Create SkyCoord object from the RA and Dec of the selected Gaia source ID

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

        print(f"\nGAIA {self.gaia_release} source ID = {gaia_result['source_id'][0]}")
        print(f"Reference epoch = {gaia_result['ref_epoch'][0]}")

        print(
            f"Parallax = {gaia_result['parallax'][0]:.2f} "
            f"+/- {gaia_result['parallax_error'][0]:.2f} mas"
        )

        print(
            f"\nRA = {gaia_result['ra'][0]:.6f} deg +/- {gaia_result['ra_error'][0]:.4f} mas"
        )
        print(
            f"Dec = {gaia_result['dec'][0]:.6f} deg +/- {gaia_result['dec_error'][0]:.4f} mas"
        )
        print(f"Coordinates = {coord_str}")

        print(
            f"\nProper motion RA = {gaia_result['pmra'][0]:.2f} "
            f"+/- {gaia_result['pmra_error'][0]:.2f} mas/yr"
        )

        print(
            f"Proper motion Dec = {gaia_result['pmdec'][0]:.2f} "
            f"+/- {gaia_result['pmdec_error'][0]:.2f} mas/yr"
        )

        if "radial_velocity" in gaia_result.columns and not np.ma.is_masked(
            gaia_result["radial_velocity"]
        ):
            print(
                f"Radial velocity = {gaia_result['radial_velocity'][0]:.2f} "
                f"+/- {gaia_result['radial_velocity_error'][0]:.2f} km/s"
            )

        print(
            f"\nG mag = {gaia_result['phot_g_mean_mag'][0]:.6f} +/- {mag_g_error:.6f}"
        )
        print(
            f"BP mag = {gaia_result['phot_bp_mean_mag'][0]:.6f} +/- {mag_bp_error:.6f}"
        )
        print(
            f"RP mag = {gaia_result['phot_rp_mean_mag'][0]:.6f} +/- {mag_rp_error:.6f}"
        )

        if "grvs_mag" in gaia_result.columns:
            print(
                f"GRVS mag = {gaia_result['grvs_mag'][0]:.6f} "
                f"+/- {gaia_result['grvs_mag_error'][0]:.6f}"
            )

        if "teff_gspphot" in gaia_result.columns and not np.ma.is_masked(
            gaia_result["teff_gspphot"]
        ):
            print(f"\nEffective temperature = {gaia_result['teff_gspphot'][0]:.0f} K")
            print(f"Surface gravity = {gaia_result['logg_gspphot'][0]:.2f}")
            print(f"Metallicity = {gaia_result['mh_gspphot'][0]:.2f}")
            print(f"G-band extinction = {gaia_result['ag_gspphot'][0]:.2f}")

        if "non_single_star" in gaia_result.columns:
            print(f"\nNon single star = {gaia_result['non_single_star'][0]}")

        if "classprob_dsc_combmod_star" in gaia_result.columns:
            print(
                "Single star probability from DSC-Combmod = "
                f"{gaia_result['classprob_dsc_combmod_star'][0]:.2f}"
            )

        print(
            f"Astrometric excess noise = {gaia_result['astrometric_excess_noise'][0]:.2f}"
        )

        if "has_xp_continuous" in gaia_result.columns:
            print(f"\nXP continuous = {gaia_result['has_xp_continuous'][0]}")

        if "has_xp_sampled" in gaia_result.columns:
            print(f"XP sampled = {gaia_result['has_xp_sampled'][0]}")

        if "has_rvs" in gaia_result.columns:
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

        # Simbad query for selected Gaia source ID

        print("\n-> Querying Simbad...\n")
        simbad_result = Simbad.query_object(
            f"GAIA {self.gaia_release} {self.gaia_source}"
        )
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

        target_dict["2MASS/2MASS.J"] = (
            float(simbad_result["FLUX_J"]),
            float(simbad_result["FLUX_ERROR_J"]),
        )

        target_dict["2MASS/2MASS.H"] = (
            float(simbad_result["FLUX_H"]),
            float(simbad_result["FLUX_ERROR_H"]),
        )

        target_dict["2MASS/2MASS.Ks"] = (
            float(simbad_result["FLUX_K"]),
            float(simbad_result["FLUX_ERROR_K"]),
        )

        # VizieR query for the selected Gaia source ID
        # Sort the result by distance from the queried object

        print("\n-> Querying VizieR...\n")

        vizier_obj = Vizier(columns=["*", "+_r"], catalog="II/328/allwise")

        radius = u.Quantity(1.0 * u.arcmin)

        vizier_result = vizier_obj.query_object(
            f"GAIA {self.gaia_release} {self.gaia_source}", radius=radius
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

        target_dict["WISE/WISE.W1"] = (
            float(vizier_result["W1mag"]),
            float(vizier_result["e_W4mag"]),
        )
        target_dict["WISE/WISE.W2"] = (
            float(vizier_result["W2mag"]),
            float(vizier_result["e_W4mag"]),
        )
        target_dict["WISE/WISE.W3"] = (
            float(vizier_result["W3mag"]),
            float(vizier_result["e_W4mag"]),
        )
        target_dict["WISE/WISE.W4"] = (
            float(vizier_result["W4mag"]),
            float(vizier_result["e_W4mag"]),
        )

        if write_json:
            json_file = f"target_{self.gaia_release.lower()}_{self.gaia_source}.json"
            print(f"\nStoring output: {json_file}")

            with open(json_file, "w", encoding="utf-8") as open_file:
                json.dump(target_dict, open_file, indent=4)

        return target_dict

    @typechecked
    def find_calib(
        self,
        search_radius: float = 0.1,
        g_mag_range: Optional[Tuple[float, float]] = None,
        write_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Function for finding calibration stars. The function returns a
        ``DataFrame`` with the sources that are queried from the Gaia
        catalog, but it also (optionally) stores the data in a CSV file
        in the working folder. The table also contains 2MASS and WISE
        magnitudes, and data from The Washington Visual Double Star
        Catalog. It is recommended to open the CSV file in a
        spreadsheet editor for easy visualization.

        Parameters
        ----------
        search_radius : float
            Radius (in degrees) of the cone that is used to query the
            GAIA source catalog to search for calibration sources in
            the vicinity of the selected ``gaia_source`` (default:
            0.1). The data release of the Gaia source catalog that is
            used for the query can be set with the ``gaia_release``
            argument of the :class:`~calistar.calistar.CaliStar`
            instance.
        g_mag_range : tuple(float, float), None
            Magnitude range relative to the the Gaia $G$ band of the
            magnitude of the selected ``gaia_source``. The magnitude
            range will be used for querying sources in the Gaia
            catalog. The argument should be specified, for example,
            as ``(-2.0, 5.0)`` if source are selected with a $G$
            magnitude that is at most 2 mag smaller and 5 mag larger
            than the magnitude ``gaia_source``.  A range of
            :math:`\\pm` 1.0 mag (i.e. ``g_mag_range=(-1.0, 1.0)``)
            is used if the argument of ``g_mag_range`` is set to
            ``None``.
        write_csv : bool
            Write the table with found source to a CSV file (default:
            True). The file will be stored in the working folder and
            starts with ``calib_find_``. The filename contains also
            the Gaia release and the Gaia source ID of the target.

        Returns
        -------
        pandas.DataFrame
            A ``DataFrame`` with the table of queried sources.
        """

        json_file = Path(f"target_{self.gaia_release.lower()}_{self.gaia_source}.json")

        if not json_file.exists():
            self.target_star(write_json=True)

        print("\n-> Finding calibration stars...\n")

        with open(json_file, "r", encoding="utf-8") as open_file:
            target_dict = json.load(open_file)

        if g_mag_range is None:
            g_mag_range = (-1.0, 1.0)

        # Add 2MASS JHKs magnitudes to the Simbad output

        # print(Simbad.list_votable_fields())

        Simbad.add_votable_fields(
            "sptype",
            "ids",
            "flux(J)",
            "flux(H)",
            "flux(K)",
            "flux_error(J)",
            "flux_error(H)",
            "flux_error(K)",
        )

        print(f"Radius of search cone = {search_radius} deg")

        mag_low = target_dict[f"GAIA/GAIA{self.gaia_idx}.G"][0] + g_mag_range[0]
        mag_upp = target_dict[f"GAIA/GAIA{self.gaia_idx}.G"][0] + g_mag_range[1]

        print(f"G mag search range = ({mag_low:.2f}, {mag_upp:.2f})")

        gaia_query = f"""
        SELECT *, DISTANCE({target_dict['Gaia RA'][0]},
        {target_dict['Gaia Dec'][0]}, ra, dec) AS ang_sep
        FROM gaiadr3.gaia_source
        WHERE DISTANCE({target_dict['Gaia RA'][0]},
        {target_dict['Gaia Dec'][0]}, ra, dec) < {search_radius}
        AND phot_g_mean_mag > {mag_low}
        AND phot_g_mean_mag < {mag_upp}
        AND parallax IS NOT NULL
        ORDER BY ang_sep ASC
        """

        # Launch the Gaia job and get the results

        gaia_job = Gaia.launch_job_async(gaia_query, dump_to_file=False, verbose=False)
        gaia_results = gaia_job.get_results()
        print(f"Number of found sources: {len(gaia_results)}")

        columns = [
            "Simbad ID",
            "Gaia ID",
            "SpT",
            "Separation",
            "Non single star",
            "Single star probability",
            "Astrometric excess noise",
        ]

        columns += self.all_filters

        columns += [
            "WDS ID",
            "WDS epoch 1",
            "WDS epoch 2",
            "WDS sep 1",
            "WDS sep 2",
            "WDS PA 1",
            "WDS PA 2",
            "WDS mag 1",
            "WDS mag 2",
        ]

        # Initiate all values in the dataframe to NaN
        cal_df = pd.DataFrame(index=range(len(gaia_results)), columns=columns)

        drop_indices = []

        warnings.filterwarnings("ignore", category=UserWarning)

        vizier_obj = Vizier(columns=["*", "+_r"])

        for gaia_item in track(gaia_results, description="Processing..."):
            cal_df.loc[gaia_item.index, "Gaia ID"] = gaia_item["source_id"]

            coord_target = SkyCoord(
                target_dict["Gaia RA"][0],
                target_dict["Gaia Dec"][0],
                frame="icrs",
                unit=(u.deg, u.deg),
            )

            coord_calib = SkyCoord(
                gaia_item["ra"],
                gaia_item["dec"],
                frame="icrs",
                unit=(u.deg, u.deg),
            )

            separation = coord_target.separation(coord_calib)

            cal_df.loc[gaia_item.index, "Separation"] = separation.deg

            cal_df.loc[gaia_item.index, f"GAIA/GAIA{self.gaia_idx}.G"] = gaia_item[
                "phot_g_mean_mag"
            ]

            simbad_result = Simbad.query_object(
                f"GAIA {self.gaia_release} {gaia_item['source_id']}"
            )

            if simbad_result is not None:
                simbad_result = simbad_result[0]

                if np.ma.is_masked(simbad_result["MAIN_ID"]):
                    cal_df.loc[gaia_item.index, "Simbad ID"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "Simbad ID"] = simbad_result["MAIN_ID"]

                if np.ma.is_masked(simbad_result["SP_TYPE"]):
                    cal_df.loc[gaia_item.index, "SpT"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "SpT"] = simbad_result["SP_TYPE"]

                if np.ma.is_masked(simbad_result["FLUX_J"]):
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.J"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.J"] = simbad_result[
                        "FLUX_J"
                    ]

                if np.ma.is_masked(simbad_result["FLUX_H"]):
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.H"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.H"] = simbad_result[
                        "FLUX_H"
                    ]

                if np.ma.is_masked(simbad_result["FLUX_K"]):
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.Ks"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.Ks"] = simbad_result[
                        "FLUX_K"
                    ]

            radius = u.Quantity(1.0 * u.arcmin)

            vizier_result = vizier_obj.query_object(
                f"GAIA {self.gaia_release} {gaia_item['source_id']}",
                radius=radius,
                catalog="II/328/allwise",
            )

            if len(vizier_result) == 1:
                vizier_result = vizier_result["II/328/allwise"][0]

                # Check if the separation between the Gaia and
                # the ALLWISE coordinates is at most 200 mas
                skip_source = vizier_result["_r"] > 0.2

                if skip_source or np.ma.is_masked(vizier_result["W1mag"]):
                    cal_df.loc[gaia_item.index, "WISE/WISE.W1"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "WISE/WISE.W1"] = vizier_result["W1mag"]

                if skip_source or np.ma.is_masked(vizier_result["W2mag"]):
                    cal_df.loc[gaia_item.index, "WISE/WISE.W2"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "WISE/WISE.W2"] = vizier_result["W2mag"]

                if skip_source or np.ma.is_masked(vizier_result["W3mag"]):
                    cal_df.loc[gaia_item.index, "WISE/WISE.W3"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "WISE/WISE.W3"] = vizier_result["W3mag"]

                if skip_source or np.ma.is_masked(vizier_result["W4mag"]):
                    cal_df.loc[gaia_item.index, "WISE/WISE.W4"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "WISE/WISE.W4"] = vizier_result["W4mag"]

                # This query returns no sources?
                # gaia_query = f"""
                # SELECT *
                # FROM gaiadr3.allwise_best_neighbour
                # WHERE source_id = {self.gaia_source}
                # """
                # gaia_job = Gaia.launch_job_async(gaia_query,
                # dump_to_file=False, verbose=False)
                # gaia_result = gaia_job.get_results()

            else:
                drop_indices.append(gaia_item.index)

            cal_df.loc[gaia_item.index, "Non single star"] = gaia_item[
                "non_single_star"
            ]

            cal_df.loc[gaia_item.index, "Single star probability"] = gaia_item[
                "classprob_dsc_combmod_star"
            ]

            cal_df.loc[gaia_item.index, "Astrometric excess noise"] = gaia_item[
                "astrometric_excess_noise"
            ]

            # Query The Washington Visual Double Star Catalog

            if simbad_result is not None:
                simbad_ids = simbad_result["IDS"].split("|")
                wds_id = list(filter(lambda x: x.startswith("WDS"), simbad_ids))

                if len(wds_id) == 1:
                    wds_table = Table.read(self.wds_file, path="wds_catalog")

                    id_crop = wds_id[0].split(" ")[-1][1:11]
                    id_idx = np.where(id_crop == wds_table["WDS"])[0]

                    if len(id_idx) == 1:
                        wds_select = wds_table[id_idx]

                        cal_df.loc[gaia_item.index, "WDS ID"] = wds_id[0]
                        cal_df.loc[gaia_item.index, "WDS epoch 1"] = wds_select["Obs1"]
                        cal_df.loc[gaia_item.index, "WDS epoch 2"] = wds_select["Obs2"]
                        cal_df.loc[gaia_item.index, "WDS sep 1"] = wds_select["sep1"]
                        cal_df.loc[gaia_item.index, "WDS sep 2"] = wds_select["sep2"]
                        cal_df.loc[gaia_item.index, "WDS PA 1"] = wds_select["pa1"]
                        cal_df.loc[gaia_item.index, "WDS PA 2"] = wds_select["pa2"]
                        cal_df.loc[gaia_item.index, "WDS mag 1"] = wds_select["mag1"]
                        cal_df.loc[gaia_item.index, "WDS mag 2"] = wds_select["mag2"]

        warnings.filterwarnings("default", category=UserWarning)

        cal_df = cal_df.drop(index=drop_indices)
        cal_df["Gaia ID"] = cal_df["Gaia ID"].astype("int")

        if write_csv:
            output_file = (
                f"calib_find_{self.gaia_release.lower()}_{self.gaia_source}.csv"
            )

            print(f"Storing output: {output_file}")

            cal_df.to_csv(path_or_buf=output_file, header=True, index=False)

        return cal_df

    @typechecked
    def select_calib(
        self,
        filter_names: Optional[List[str]] = None,
        mag_diff: Union[float, Dict[str, float]] = 0.1,
        write_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Function for selecting the calibration stars. The function
        returns a ``DataFrame`` with the selected sources, but it also
        (optionally) stores the data in a CSV file in the working
        folder. It is recommended to open the CSV file in a spreadsheet
        editor for easy visualization.

        Parameters
        ----------
        filter_names : list(str), None
            List with filter names that are used in combination
            with ``mag_diff`` for selecting sources. Any of the 2MASS,
            WISE, and GAIA filter names from the `SVO Filter Profile
            Service <http://svo2.cab.inta-csic.es/theory/fps/>`_ can
            be used. (default: ``['2MASS/2MASS.J', '2MASS/2MASS.H',
            '2MASS/2MASS.Ks']``).
        mag_diff : float, dict(str, float)
            Allowed magnitude difference between the selected target
            (i.e. the argument of ``gaia_source``) and the sources
            there were found with
            :func:`~calistar.calistar.CaliStar.find_calib()` (default:
            0.1). The argument can be either a float, in which case the
            same value is used for all filters listed in
            ``filter_names``, or a dictionary in which case the keys
            should be the filter names that are listed in
            ``filter_names`` and the values are the allowed magnitude
            differences for each filter.
        write_csv : bool
            Write the table with found source to a CSV file (default:
            True). The file will be stored in the working folder and
            starts with ``calib_select_``. The filename contains also
            the Gaia release and the Gaia source ID of the target.

        Returns
        -------
        pandas.DataFrame
            The ``DataFrame`` with the selected calibration stars.
        """

        json_file = Path(f"target_{self.gaia_release.lower()}_{self.gaia_source}.json")

        if not json_file.exists():
            self.target_star(write_json=True)

        print("\n-> Selecting calibration stars...\n")

        with open(json_file, "r", encoding="utf-8") as open_file:
            target_dict = json.load(open_file)

        if filter_names is None:
            filter_names = ["2MASS/2MASS.J", "2MASS/2MASS.H", "2MASS/2MASS.Ks"]

        if not isinstance(mag_diff, dict):
            diff_val = copy(mag_diff)
            mag_diff = {}

            for filter_item in filter_names:
                mag_diff[filter_item] = diff_val

        if sorted(filter_names) != sorted(list(mag_diff.keys())):
            raise ValueError(
                "The values in the list of 'filter_names', "
                f"{filter_names}, is not equal to the keys in the "
                f"dictionary of 'mag_diff', {list(mag_diff.keys())}."
            )

        csv_file = Path(
            f"calib_find_{self.gaia_release.lower()}_{self.gaia_source}.csv"
        )

        if not csv_file.exists():
            err_msg = (
                "The CSV file with pre-selected calibration "
                "sources is not found. Please make sure to run "
                "the 'find_calib()' method before "
                "'select_calib()', and set the argument of "
                "'write_csv' to True."
            )

            raise FileNotFoundError(err_msg)

        cal_df = pd.read_csv(
            filepath_or_buffer=csv_file,
            header=0,
            index_col=False,
        )

        drop_indices = []

        for row_idx in track(range(len(cal_df)), description="Processing..."):
            for filter_item in filter_names:
                if np.isnan(cal_df.loc[row_idx, filter_item]) or (
                    np.abs(
                        cal_df.loc[row_idx, filter_item] - target_dict[filter_item][0]
                    )
                    > mag_diff[filter_item]
                ):
                    if row_idx not in drop_indices:
                        drop_indices.append(row_idx)

        cal_df = cal_df.drop(index=drop_indices)

        print(f"Number of selected sources: {len(cal_df)}")

        if write_csv:
            output_file = (
                f"calib_select_{self.gaia_release.lower()}_{self.gaia_source}.csv"
            )

            print(f"Storing output: {output_file}")

            cal_df.to_csv(path_or_buf=output_file, header=True, index=False)

        return cal_df
