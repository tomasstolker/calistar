"""
Module with the ``calistar`` tool.
"""

import json
import socket
import urllib.request
import warnings

from copy import copy
from pathlib import Path

import astropy.units as u
from beartype import beartype, typing
import numpy as np
import pandas as pd
import pooch

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from gaiaxpy import calibrate, plot_spectra

# from gaiaxpy.calibrator.calibrator import __create_merge as create_merge
from gaiaxpy.core.generic_functions import correlation_to_covariance
from tqdm import tqdm

from ._version import __version__, __version_tuple__


# No limit on the number of rows with a Gaia query
Gaia.ROW_LIMIT = -1


class CaliStar:
    """
    Class for finding calibration stars based on their separation
    and magnitude difference with the selected ``gaia_source``.
    """

    @beartype
    def __init__(
        self,
        gaia_source: typing.Union[int, str],
        gaia_release: typing.Optional[str] = None,
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

        print("========\ncalistar\n========")

        # Check if there is a new version available

        print(f"\nVersion: {__version__}")

        calistar_version = (
            f"{__version_tuple__[0]}."
            f"{__version_tuple__[1]}."
            f"{__version_tuple__[2]}"
        )

        try:
            pypi_url = "https://pypi.org/pypi/calistar/json"

            with urllib.request.urlopen(pypi_url, timeout=1.0) as open_url:
                url_content = open_url.read()
                url_data = json.loads(url_content)
                pypi_version = url_data["info"]["version"]

        except (urllib.error.URLError, socket.timeout):
            pypi_version = None

        if pypi_version is not None:
            pypi_split = pypi_version.split(".")
            current_split = calistar_version.split(".")

            new_major = (pypi_split[0] == current_split[0]) & (
                pypi_split[1] > current_split[1]
            )

            new_minor = (
                (pypi_split[0] == current_split[0])
                & (pypi_split[1] == current_split[1])
                & (pypi_split[2] > current_split[2])
            )

            if new_major | new_minor:
                print(f"\n-> calistar v{pypi_version} is available!")

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

    @beartype
    def target_star(
        self,
        write_json: bool = True,
        get_gaiaxp: bool = True,
        allwise_catalog: bool = True,
        print_astroph: bool = False,
    ) -> typing.Dict[str, typing.Union[float, str, int, typing.Tuple[float, float]]]:
        """
        Function for retrieving the the astrometric and
        photometric properties of a target star of interest. The
        function returns a dictionary with the properties, but it
        also (optionally) stores the data in a JSON file in the
        working folder and retrieves the Gaia XP spectrum.

        Parameters
        ----------
        write_json : bool
            Write the target properties to a JSON file (default: True).
            The file will be stored in the working folder and starts
            with ``target_``. The filename contains also the Gaia
            release and the Gaia source ID of the target.
        get_gaiaxp : bool
            Retrieve the Gaia XP spectrum if available (default: True).
            If set to ``True``, the spectrum will be written to a data
            file and a plot will also be created. The spectrum is not
            retrieved when the argument is set to ``False``.
        allwise_catalog : bool
            Select the WISE magnitudes from the ALLWISE catalog if set
            to ``True`` or select the magnitudes from the earlier WISE
            catalog if set to ``False``.
        print_astroph : bool
            Print a list with all the astrophysical parameters that
            will be retrieved from the Gaia catalog when the
            argument is set to ``True``.

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

        else:
            raise ValueError(
                f"The '{self.gaia_release}' data release is not supported."
            )

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

        if "SOURCE_ID" in gaia_result.columns:
            gaia_source_id = int(gaia_result["SOURCE_ID"][0])
        elif "source_id" in gaia_result.columns:
            gaia_source_id = int(gaia_result["source_id"][0])
        else:
            raise ValueError(f"Gaia source ID not found in {gaia_result}")

        target_dict["Gaia ID"] = gaia_source_id

        target_dict["Gaia release"] = self.gaia_release

        target_dict["Gaia epoch"] = gaia_result["ref_epoch"][0]

        target_dict["Gaia RA"] = (
            float(gaia_result["ra"][0]),  # (deg)
            float(gaia_result["ra_error"][0] * 1e-3 / 3600.0),  # (mas) -> (deg)
        )

        target_dict["Gaia Dec"] = (
            float(gaia_result["dec"][0]),  # (deg)
            float(gaia_result["dec_error"][0] * 1e-3 / 3600.0),  # (mas) -> (deg)
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

        if "phot_g_mean_mag" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["phot_g_mean_mag"][0]):
                target_dict[f"GAIA/GAIA{self.gaia_idx}.G"] = (
                    float(gaia_result["phot_g_mean_mag"][0]),
                    mag_g_error,
                )

                print(
                    f"\nG mag = {gaia_result['phot_g_mean_mag'][0]:.6f} +/- {mag_g_error:.6f}"
                )

        if "phot_bp_mean_mag" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["phot_bp_mean_mag"][0]):
                target_dict[f"GAIA/GAIA{self.gaia_idx}.Gbp"] = (
                    float(gaia_result["phot_bp_mean_mag"][0]),
                    mag_bp_error,
                )

                print(
                    f"BP mag = {gaia_result['phot_bp_mean_mag'][0]:.6f} +/- {mag_bp_error:.6f}"
                )

        if "phot_rp_mean_mag" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["phot_rp_mean_mag"][0]):
                target_dict[f"GAIA/GAIA{self.gaia_idx}.Grp"] = (
                    float(gaia_result["phot_rp_mean_mag"][0]),
                    mag_rp_error,
                )

                print(
                    f"RP mag = {gaia_result['phot_rp_mean_mag'][0]:.6f} +/- {mag_rp_error:.6f}"
                )

        if "grvs_mag" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["grvs_mag"][0]):
                target_dict[f"GAIA/GAIA{self.gaia_idx}.Grvs"] = (
                    float(gaia_result["grvs_mag"][0]),
                    float(gaia_result["grvs_mag_error"][0]),
                )

                print(
                    f"GRVS mag = {gaia_result['grvs_mag'][0]:.6f} "
                    f"+/- {gaia_result['grvs_mag_error'][0]:.6f}"
                )

        # Create SkyCoord object from the RA and Dec of the selected Gaia source ID

        gaia_coord = SkyCoord(
            gaia_result["ra"][0],
            gaia_result["dec"][0],
            frame="icrs",
            unit=(u.deg, u.deg),
        )

        coord_str = gaia_coord.to_string(
            "hmsdms", alwayssign=True, precision=4, pad=True
        )

        if self.gaia_source != gaia_source_id:
            raise ValueError(
                f"The requested source ID ({self.gaia_source}) is not "
                f"equal to the retrieved source ID ({gaia_source_id})."
            )

        print(f"\nGAIA {self.gaia_release} source ID = {gaia_source_id}")
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

        if "radial_velocity" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["radial_velocity"]):
                print(
                    f"Radial velocity = {gaia_result['radial_velocity'][0]:.2f} "
                    f"+/- {gaia_result['radial_velocity_error'][0]:.2f} km/s"
                )

        print(
            f"\nAstrometric excess noise = {gaia_result['astrometric_excess_noise'][0]:.2f}"
        )

        if "ruwe" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["ruwe"]):
                print(f"RUWE = {gaia_result['ruwe'][0]:.2f}")

        if "non_single_star" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["non_single_star"]):
                print(f"Non single star = {gaia_result['non_single_star'][0]}")

        if "classprob_dsc_combmod_star" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["classprob_dsc_combmod_star"]):
                print(
                    "Single star probability from DSC-Combmod = "
                    f"{gaia_result['classprob_dsc_combmod_star'][0]:.2f}"
                )

        if "has_xp_continuous" in gaia_result.columns:
            print(f"\nXP continuous = {gaia_result['has_xp_continuous'][0]}")

        if "has_xp_sampled" in gaia_result.columns:
            print(f"XP sampled = {gaia_result['has_xp_sampled'][0]}")

        if "has_rvs" in gaia_result.columns:
            print(f"RVS spectrum = {gaia_result['has_rvs'][0]}")

        if "teff_gspphot" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["teff_gspphot"]):
                print(
                    f"\nEffective temperature = {gaia_result['teff_gspphot'][0]:.0f} K"
                )
                print(f"Surface gravity = {gaia_result['logg_gspphot'][0]:.2f}")
                print(f"Metallicity = {gaia_result['mh_gspphot'][0]:.2f}")
                print(f"G-band extinction = {gaia_result['ag_gspphot'][0]:.2f}")
                print(
                    f"A0 (541.4 nm) extinction = {gaia_result['azero_gspphot'][0]:.2f}"
                )

                target_dict["teff"] = float(gaia_result["teff_gspphot"][0])
                target_dict["log(g)"] = float(gaia_result["logg_gspphot"][0])
                target_dict["metallicity"] = float(gaia_result["mh_gspphot"][0])
                target_dict["ag_ext"] = float(gaia_result["ag_gspphot"][0])
                target_dict["azero_ext"] = float(gaia_result["azero_gspphot"][0])

        # Gaia query for selected the astrophysical parameters

        if print_astroph:
            print("\n-> Querying astrophysical parameters...\n")

            gaia_query = f"""
            SELECT *
            FROM gaia{self.gaia_release.lower()}.astrophysical_parameters
            WHERE source_id = {self.gaia_source}
            """

            # Launch the Gaia job and get the results

            gaia_job = Gaia.launch_job_async(
                gaia_query, dump_to_file=False, verbose=False
            )
            gaia_result = gaia_job.get_results()

            if len(gaia_result) == 0:
                print("\nTarget has no data in the astrophysical_parameters catalog")

            else:
                print("\nAstrophysical parameters:")

                for param_item in gaia_result.columns:
                    print(f"   - {param_item} = {gaia_result.columns[param_item][0]}")

        # Gaia XP spectrum

        if get_gaiaxp and (
            "has_xp_continuous" in gaia_result.columns
            and gaia_result["has_xp_continuous"][0]
        ):
            # Sampling adopted from the GaiaXPy documentation
            # https://gaiaxpy.readthedocs.io/en/latest/usage.html

            # Default GaiaXPy sampling
            # sampling = np.arange(336, 1021, 2)

            # Improved sampling at the blue end of the spectrum
            sampling = np.geomspace(330, 1049.9999999999, 361)

            df_cal, sampling = calibrate(
                input_object=[f"{self.gaia_source}"],
                sampling=sampling,
                truncation=False,
                with_correlation=True,
                output_path="./",
                # output_file=f"{self.gaia_source}_gaiaxp",
                output_format=None,
                save_file=False,
                username=None,
                password=None,
            )

            # merge_bp = create_merge(xp='bp', sampling=sampling)
            # merge_rp = create_merge(xp='rp', sampling=sampling)

            xp_plot = f"gaiaxp_{self.gaia_source}"
            print(f"\nStoring Gaia XP plot: {xp_plot}_0.jpg")

            plot_spectra(
                spectra=df_cal,
                sampling=sampling,
                multi=False,
                show_plot=False,
                output_path="./",
                output_file=xp_plot,
                format=None,
                legend=True,
            )

            xp_wavel = sampling * 1e-3  # (nm) -> (um)
            xp_flux = 1e3 * df_cal["flux"][0]  # (W m-2 nm-1) -> (W m-2 um-1)
            xp_error = 1e3 * df_cal["flux_error"][0]

            xp_cov = correlation_to_covariance(
                correlation=df_cal["correlation"][0],
                error=df_cal["flux_error"][0],
                stdev=1.0,
            )

            xp_cov *= 1e6  # (W m-2 nm-1)^2 -> (W m-2 um-1)^2

            header = "Wavelength (um) - Flux (W m-2 um-1) - Uncertainty (W m-2 um-1)"
            xp_spec_file = f"gaiaxp_{self.gaia_source}_spec.dat"
            xp_spec = np.column_stack([xp_wavel, xp_flux, xp_error])
            np.savetxt(xp_spec_file, xp_spec, header=header)
            print(f"Storing Gaia XP spectrum: {xp_spec_file}")

            header = "Covariances (W m-2 um-1)^2"
            xp_cov_file = f"gaiaxp_{self.gaia_source}_cov.dat"
            np.savetxt(xp_cov_file, xp_cov, header=header)
            print(f"Storing Gaia XP covariances: {xp_cov_file}")

        # Add spectral type to the Simbad output

        # for item in Simbad.list_votable_fields():
        #     print(item)

        Simbad.add_votable_fields(
            "sp_type",
            "ids",
            "sp",
            "sp_qual",
            "sp_bibcode",
            "otype",
            "otype_txt",
            # "flux(J)",
            # "flux(H)",
            # "flux(K)",
            # "flux_error(J)",
            # "flux_error(H)",
            # "flux_error(K)",
        )

        # Simbad query for selected Gaia source ID

        print("\n-> Querying Simbad...\n")

        simbad_result = Simbad.query_object(
            f"GAIA {self.gaia_release} {self.gaia_source}"
        )

        if simbad_result is not None and len(simbad_result) > 0:
            simbad_result = simbad_result[0]

            print(f"Simbad ID = {simbad_result['main_id']}")
            print(f"Object type = {simbad_result['otype_txt']}")
            print(f"Spectral type = {simbad_result['sp_type']}")
            print(f"Reference = {simbad_result['sp_bibcode']}")

            target_dict["Simbad ID"] = simbad_result["main_id"]
            target_dict["Spectral Type"] = simbad_result["sp_type"]
            target_dict["Object Type"] = simbad_result["otype_txt"]

            # print(
            #     f"\n2MASS J mag = {simbad_result['FLUX_J']:.3f} "
            #     f"+/- {simbad_result['FLUX_ERROR_J']:.3f}"
            # )
            #
            # print(
            #     f"2MASS H mag = {simbad_result['FLUX_H']:.3f} "
            #     f"+/- {simbad_result['FLUX_ERROR_H']:.3f}"
            # )
            #
            # print(
            #     f"2MASS Ks mag = {simbad_result['FLUX_K']:.3f} "
            #     f"+/- {simbad_result['FLUX_ERROR_K']:.3f}"
            # )

            # target_dict["2MASS/2MASS.J"] = (
            #     float(simbad_result["FLUX_J"]),
            #     float(simbad_result["FLUX_ERROR_J"]),
            # )
            #
            # target_dict["2MASS/2MASS.H"] = (
            #     float(simbad_result["FLUX_H"]),
            #     float(simbad_result["FLUX_ERROR_H"]),
            # )
            #
            # target_dict["2MASS/2MASS.Ks"] = (
            #     float(simbad_result["FLUX_K"]),
            #     float(simbad_result["FLUX_ERROR_K"]),
            # )

        else:
            print("\nTarget not found on Simbad")

        # VizieR query for the selected Gaia source ID
        # Sort the result by distance from the queried object

        print("\n-> Querying VizieR...\n")

        vizier_obj = Vizier(
            columns=["*", "+_r", "BTmag", "e_BTmag", "VTmag", "e_VTmag"],
            catalog=["I/259/tyc2", "II/246/out", "II/328/allwise", "II/311/wise"],
            timeout=10.0,
            row_limit=1,
        )

        radius = u.Quantity(1.0 * u.arcmin)

        vizier_result = vizier_obj.query_object(
            f"GAIA {self.gaia_release} {self.gaia_source}", radius=radius
        )

        # TYCHO data from VizieR

        if "I/259/tyc2" in vizier_result.keys():
            vizier_tycho = vizier_result["I/259/tyc2"]
        else:
            vizier_tycho = None

        if vizier_tycho is not None and len(vizier_tycho) > 0:
            vizier_tycho = vizier_tycho[0]

            print(
                f"TYCHO source ID = {vizier_tycho['TYC1']}-"
                f"{vizier_tycho['TYC2']}-{vizier_tycho['TYC3']}"
            )

            print(
                f"Separation between Gaia and TYCHO source = "
                f"{1e3*vizier_tycho['_r']:.1f} mas"
            )

            target_dict["TYCHO separation"] = 1e3 * vizier_tycho["_r"]

            if 1e3 * vizier_tycho["_r"] > 10.0:
                warnings.warn(
                    "The separation between the Gaia and TYCHO source "
                    "is more than 10 mas. Please check carefully if "
                    "these are indeed the same sources."
                )

            if np.ma.is_masked(vizier_tycho["e_BTmag"]):
                if not np.ma.is_masked(vizier_tycho["BTmag"]):
                    print(f"\nTYCHO BT mag = >{vizier_tycho['BTmag']:.3f}")

            else:
                print(
                    f"\nTYCHO BT mag = {vizier_tycho['BTmag']:.3f} "
                    f"+/- {vizier_tycho['e_BTmag']:.3f}"
                )

                target_dict["TYCHO/TYCHO.B"] = (
                    float(vizier_tycho["BTmag"]),
                    float(vizier_tycho["e_BTmag"]),
                )

            if np.ma.is_masked(vizier_tycho["e_VTmag"]):
                if not np.ma.is_masked(vizier_tycho["VTmag"]):
                    print(f"\nTYCHO VT mag = >{vizier_tycho['VTmag']:.3f}")

            else:
                print(
                    f"TYCHO VT mag = {vizier_tycho['VTmag']:.3f} "
                    f"+/- {vizier_tycho['e_VTmag']:.3f}"
                )

                target_dict["TYCHO/TYCHO.V"] = (
                    float(vizier_tycho["VTmag"]),
                    float(vizier_tycho["e_VTmag"]),
                )

        else:
            print("Target not found in TYCHO catalog")

        # 2MASS data from VizieR

        if "II/246/out" in vizier_result.keys():
            vizier_2mass = vizier_result["II/246/out"]
        else:
            vizier_2mass = None

        if vizier_2mass is not None and len(vizier_2mass) > 0:
            vizier_2mass = vizier_2mass[0]

            print(f"\n2MASS source ID = {vizier_2mass['2MASS']}")

            target_dict["2MASS ID"] = vizier_2mass["2MASS"]

            print(
                f"Separation between Gaia and 2MASS source = "
                f"{1e3*vizier_2mass['_r']:.1f} mas"
            )

            target_dict["2MASS separation"] = 1e3 * vizier_2mass["_r"]

            if 1e3 * vizier_2mass["_r"] > 10.0:
                warnings.warn(
                    "The separation between the Gaia and 2MASS source "
                    "is more than 10 mas. Please check carefully if "
                    "these are indeed the same sources."
                )

            if np.ma.is_masked(vizier_2mass["e_Jmag"]):
                if not np.ma.is_masked(vizier_2mass["Jmag"]):
                    print(f"\n2MASS J mag = >{vizier_2mass['Jmag']:.3f}")

            else:
                print(
                    f"\n2MASS J mag = {vizier_2mass['Jmag']:.3f} "
                    f"+/- {vizier_2mass['e_Jmag']:.3f}"
                )

                target_dict["2MASS/2MASS.J"] = (
                    float(vizier_2mass["Jmag"]),
                    float(vizier_2mass["e_Jmag"]),
                )

            if np.ma.is_masked(vizier_2mass["e_Hmag"]):
                if not np.ma.is_masked(vizier_2mass["Hmag"]):
                    print(f"2MASS H mag = >{vizier_2mass['Hmag']:.3f}")

            else:
                print(
                    f"2MASS H mag = {vizier_2mass['Hmag']:.3f} "
                    f"+/- {vizier_2mass['e_Hmag']:.3f}"
                )

                target_dict["2MASS/2MASS.H"] = (
                    float(vizier_2mass["Hmag"]),
                    float(vizier_2mass["e_Hmag"]),
                )

            if np.ma.is_masked(vizier_2mass["e_Kmag"]):
                if not np.ma.is_masked(vizier_2mass["Kmag"]):
                    print(f"2MASS Ks mag = >{vizier_2mass['Kmag']:.3f}")

            else:
                print(
                    f"2MASS Ks mag = {vizier_2mass['Kmag']:.3f} "
                    f"+/- {vizier_2mass['e_Kmag']:.3f}"
                )

                target_dict["2MASS/2MASS.Ks"] = (
                    float(vizier_2mass["Kmag"]),
                    float(vizier_2mass["e_Kmag"]),
                )

        else:
            print("Target not found in 2MASS catalog")

        # WISE data from VizieR

        vizier_wise = None

        if allwise_catalog:
            if "II/328/allwise" in vizier_result.keys():
                vizier_wise = vizier_result["II/328/allwise"]
        else:
            if "II/311/wise" in vizier_result.keys():
                vizier_wise = vizier_result["II/311/wise"]

        if vizier_wise is not None and len(vizier_wise) > 0:
            vizier_wise = vizier_wise[0]

            if allwise_catalog:
                print(f"\nALLWISE source ID = {vizier_wise['AllWISE']}")
                target_dict["WISE ID"] = vizier_wise["AllWISE"]

            else:
                print(f"\nWISE source ID = {vizier_wise['WISE']}")
                target_dict["WISE ID"] = vizier_wise["WISE"]

            print(
                f"Separation between Gaia and WISE source = "
                f"{1e3*vizier_wise['_r']:.1f} mas"
            )

            target_dict["WISE separation"] = 1e3 * vizier_wise["_r"]

            if 1e3 * vizier_wise["_r"] > 10.0:
                warnings.warn(
                    "The separation between the Gaia and WISE source "
                    "is more than 10 mas. Please check carefully if "
                    "these are indeed the same sources."
                )

            if np.ma.is_masked(vizier_wise["e_W1mag"]):
                if not np.ma.is_masked(vizier_wise["W1mag"]):
                    print(f"\nWISE W1 mag = >{vizier_wise['W1mag']:.3f}")

            else:
                print(
                    f"\nWISE W1 mag = {vizier_wise['W1mag']:.3f} "
                    f"+/- {vizier_wise['e_W1mag']:.3f}"
                )

                target_dict["WISE/WISE.W1"] = (
                    float(vizier_wise["W1mag"]),
                    float(vizier_wise["e_W1mag"]),
                )

            if np.ma.is_masked(vizier_wise["e_W2mag"]):
                if not np.ma.is_masked(vizier_wise["W2mag"]):
                    print(f"WISE W2 mag = >{vizier_wise['W2mag']:.3f}")

            else:
                print(
                    f"WISE W2 mag = {vizier_wise['W2mag']:.3f} "
                    f"+/- {vizier_wise['e_W2mag']:.3f}"
                )

                target_dict["WISE/WISE.W2"] = (
                    float(vizier_wise["W2mag"]),
                    float(vizier_wise["e_W2mag"]),
                )

            if np.ma.is_masked(vizier_wise["e_W3mag"]):
                if not np.ma.is_masked(vizier_wise["W3mag"]):
                    print(f"WISE W3 mag = >{vizier_wise['W3mag']:.3f}")

            else:
                print(
                    f"WISE W3 mag = {vizier_wise['W3mag']:.3f} "
                    f"+/- {vizier_wise['e_W3mag']:.3f}"
                )

                target_dict["WISE/WISE.W3"] = (
                    float(vizier_wise["W3mag"]),
                    float(vizier_wise["e_W3mag"]),
                )

            if np.ma.is_masked(vizier_wise["e_W4mag"]):
                if not np.ma.is_masked(vizier_wise["W4mag"]):
                    print(f"WISE W4 mag = >{vizier_wise['W4mag']:.3f}")

            else:
                print(
                    f"WISE W4 mag = {vizier_wise['W4mag']:.3f} "
                    f"+/- {vizier_wise['e_W4mag']:.3f}"
                )

                target_dict["WISE/WISE.W4"] = (
                    float(vizier_wise["W4mag"]),
                    float(vizier_wise["e_W4mag"]),
                )

        else:
            print("Target not found in WISE catalog")

        print("\n-> Querying Washington Double Star catalog...\n")

        found_wds = False

        if simbad_result is not None and len(simbad_result) > 0:
            simbad_ids = simbad_result["ids"].split("|")
            wds_id = list(filter(lambda x: x.startswith("WDS"), simbad_ids))

            if len(wds_id) == 1:
                # There should be a single WDS identified per target on Simbad
                wds_id = wds_id[0]

                wds_table = Table.read(self.wds_file, path="wds_catalog")
                # print(wds_table.columns)

                # This will not always given the correct ID
                # only for regular binary system IDs
                id_crop = wds_id.split(" ")[-1][1:11]
                id_idx = np.where(id_crop == wds_table["WDS"])[0]

                target_dict["WDS ID"] = wds_id

                for wds_idx in range(len(id_idx)):
                    if wds_idx > 0:
                        print()

                    wds_select = wds_table[id_idx[wds_idx]]
                    print(f"WDS ID = {wds_select['WDS']}")

                    if len(wds_select["Comp"]) > 0:
                        print(f"Companion = {wds_select['Comp']}")

                    print(f"Observation 1 = {wds_select['Obs1']}")
                    print(f"Observation 2 = {wds_select['Obs2']}")

                    if wds_select["sep1"] is not np.ma.masked:
                        print(f"Separation 1 (arcsec) = {wds_select['sep1']:.2f}")
                    else:
                        print("Separation 1 (arcsec) = --")

                    if wds_select["sep2"] is not np.ma.masked:
                        print(f"Separation 2 (arcsec) = {wds_select['sep2']:.2f}")
                    else:
                        print("Separation 2 (arcsec) = --")

                    if wds_select["pa1"] is not np.ma.masked:
                        print(f"Position angle 1 (deg) = {wds_select['pa1']:.2f}")
                    else:
                        print("Position angle 1 (deg) = --")

                    if wds_select["pa2"] is not np.ma.masked:
                        print(f"Position angle 2 (deg) = {wds_select['pa2']:.2f}")
                    else:
                        print("Position angle 2 (deg) = --")

                    if wds_select["mag1"] is not np.ma.masked:
                        print(f"Magnitude 1 = {wds_select['mag1']:.2f}")
                    else:
                        print("Magnitude 1 = --")

                    if wds_select["mag2"] is not np.ma.masked:
                        print(f"Magnitude 2 = {wds_select['mag2']:.2f}")
                    else:
                        print("Magnitude 2 = --")

                    found_wds = True

        if not found_wds:
            print("Target not found in WDS catalog")

        if write_json:
            json_file = f"target_{self.gaia_release.lower()}_{self.gaia_source}.json"

            print(f"\nStoring JSON output: {json_file}")

            with open(json_file, "w", encoding="utf-8") as open_file:
                json.dump(target_dict, open_file, indent=4)

        return target_dict

    @beartype
    def find_calib(
        self,
        search_radius: float = 0.1,
        g_mag_range: typing.Optional[typing.Tuple[float, float]] = None,
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
            "sp_type",
            "ids",
            # "flux(J)",
            # "flux(H)",
            # "flux(K)",
            # "flux_error(J)",
            # "flux_error(H)",
            # "flux_error(K)",
        )

        print(f"Radius of search cone = {search_radius} deg")

        mag_low = target_dict[f"GAIA/GAIA{self.gaia_idx}.G"][0] + g_mag_range[0]
        mag_upp = target_dict[f"GAIA/GAIA{self.gaia_idx}.G"][0] + g_mag_range[1]

        print(f"G mag search range = ({mag_low:.2f}, {mag_upp:.2f})")

        gaia_query = f"""
        SELECT *,
        DISTANCE(
            POINT('ICRS', ra, dec),
            POINT('ICRS', {target_dict['Gaia RA'][0]}, {target_dict['Gaia Dec'][0]})
        ) AS ang_sep
        FROM gaia{self.gaia_release.lower()}.gaia_source
        WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE(
                  'ICRS',
                  {target_dict['Gaia RA'][0]},
                  {target_dict['Gaia Dec'][0]},
                  {search_radius}
                )
              ) = 1
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
            "Astrometric excess noise",
            "RUWE",
            "Non single star",
            "Single star probability",
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

        vizier_obj = Vizier(columns=["*", "+_r"], timeout=10.0, row_limit=1)

        for gaia_item in tqdm(gaia_results):
            if "SOURCE_ID" in gaia_item.columns:
                gaia_source_id = int(gaia_item["SOURCE_ID"])
            elif "source_id" in gaia_item.columns:
                gaia_source_id = int(gaia_item["source_id"])
            else:
                raise ValueError(f"Gaia source ID not found in {gaia_item}")

            cal_df.loc[gaia_item.index, "Gaia ID"] = gaia_source_id

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
                f"GAIA {self.gaia_release} {gaia_source_id}"
            )

            if simbad_result is not None and len(simbad_result) > 0:
                simbad_result = simbad_result[0]

                if np.ma.is_masked(simbad_result["main_id"]):
                    cal_df.loc[gaia_item.index, "Simbad ID"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "Simbad ID"] = simbad_result["main_id"]

                if np.ma.is_masked(simbad_result["sp_type"]):
                    cal_df.loc[gaia_item.index, "SpT"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "SpT"] = simbad_result["sp_type"]

                # if np.ma.is_masked(simbad_result["FLUX_J"]):
                #     cal_df.loc[gaia_item.index, "2MASS/2MASS.J"] = np.nan
                # else:
                #     cal_df.loc[gaia_item.index, "2MASS/2MASS.J"] = simbad_result[
                #         "FLUX_J"
                #     ]
                #
                # if np.ma.is_masked(simbad_result["FLUX_H"]):
                #     cal_df.loc[gaia_item.index, "2MASS/2MASS.H"] = np.nan
                # else:
                #     cal_df.loc[gaia_item.index, "2MASS/2MASS.H"] = simbad_result[
                #         "FLUX_H"
                #     ]
                #
                # if np.ma.is_masked(simbad_result["FLUX_K"]):
                #     cal_df.loc[gaia_item.index, "2MASS/2MASS.Ks"] = np.nan
                # else:
                #     cal_df.loc[gaia_item.index, "2MASS/2MASS.Ks"] = simbad_result[
                #         "FLUX_K"
                #     ]

            radius = u.Quantity(1.0 * u.arcmin)

            vizier_result = vizier_obj.query_object(
                f"GAIA {self.gaia_release} {gaia_source_id}",
                radius=radius,
                catalog=["II/246/out", "II/328/allwise"],
            )

            if len(vizier_result) == 2:
                # 2MASS
                vizier_2mass = vizier_result["II/246/out"][0]

                # Check if the separation between the Gaia and
                # the 2MASS coordinates is at most 200 mas
                skip_source = vizier_2mass["_r"] > 0.2

                if skip_source or np.ma.is_masked(vizier_2mass["Jmag"]):
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.J"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.J"] = vizier_2mass["Jmag"]

                if skip_source or np.ma.is_masked(vizier_2mass["Hmag"]):
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.H"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.H"] = vizier_2mass["Hmag"]

                if skip_source or np.ma.is_masked(vizier_2mass["Kmag"]):
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.Ks"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "2MASS/2MASS.Ks"] = vizier_2mass["Kmag"]

                # WISE

                vizier_wise = vizier_result["II/328/allwise"][0]

                # Check if the separation between the Gaia and
                # the ALLWISE coordinates is at most 200 mas
                skip_source = vizier_wise["_r"] > 0.2

                if skip_source or np.ma.is_masked(vizier_wise["W1mag"]):
                    cal_df.loc[gaia_item.index, "WISE/WISE.W1"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "WISE/WISE.W1"] = vizier_wise["W1mag"]

                if skip_source or np.ma.is_masked(vizier_wise["W2mag"]):
                    cal_df.loc[gaia_item.index, "WISE/WISE.W2"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "WISE/WISE.W2"] = vizier_wise["W2mag"]

                if skip_source or np.ma.is_masked(vizier_wise["W3mag"]):
                    cal_df.loc[gaia_item.index, "WISE/WISE.W3"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "WISE/WISE.W3"] = vizier_wise["W3mag"]

                if skip_source or np.ma.is_masked(vizier_wise["W4mag"]):
                    cal_df.loc[gaia_item.index, "WISE/WISE.W4"] = np.nan
                else:
                    cal_df.loc[gaia_item.index, "WISE/WISE.W4"] = vizier_wise["W4mag"]

                # This query returns no sources?
                # gaia_query = f"""
                # SELECT *
                # FROM gaia{self.gaia_release.lower()}.allwise_best_neighbour
                # WHERE source_id = {self.gaia_source}
                # """
                # gaia_job = Gaia.launch_job_async(gaia_query,
                # dump_to_file=False, verbose=False)
                # gaia_result = gaia_job.get_results()

            else:
                drop_indices.append(gaia_item.index)

            cal_df.loc[gaia_item.index, "Astrometric excess noise"] = gaia_item[
                "astrometric_excess_noise"
            ]

            if "ruwe" in gaia_item:
                cal_df.loc[gaia_item.index, "RUWE"] = gaia_item["ruwe"]

            if "non_single_star" in gaia_item:
                cal_df.loc[gaia_item.index, "Non single star"] = gaia_item[
                    "non_single_star"
                ]

            if "classprob_dsc_combmod_star" in gaia_item:
                cal_df.loc[gaia_item.index, "Single star probability"] = gaia_item[
                    "classprob_dsc_combmod_star"
                ]

            # Query The Washington Visual Double Star Catalog

            if simbad_result is not None and len(simbad_result) > 0:
                simbad_ids = simbad_result["ids"].split("|")
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

    @beartype
    def select_calib(
        self,
        filter_names: typing.Optional[typing.List[str]] = None,
        mag_diff: typing.Union[float, typing.Dict[str, float]] = 0.1,
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

        for row_idx in tqdm(range(len(cal_df))):
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
