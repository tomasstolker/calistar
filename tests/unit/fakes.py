from astropy.table import Table


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
