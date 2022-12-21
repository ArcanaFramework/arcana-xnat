import os
import logging
import sys
from tempfile import mkdtemp
import json
import tempfile
from datetime import datetime
from pathlib import Path
import pytest
import numpy
import nibabel
from click.testing import CliRunner
import xnat4tests
from arcana.xnat.data import Xnat
from arcana.medimage.data import Clinical, NiftiGzX, NiftiGz, Dicom, NiftiX
from arcana.common.data import Text, Directory
from arcana.xnat.utils.testing import (
    make_mutable_dataset,
    TestXnatDatasetBlueprint,
    ResourceBlueprint,
    ScanBlueprint,
    DerivBlueprint,
    create_dataset_data_in_repo,
    make_project_id,
    access_dataset,
)
from arcana.core.utils.testing.data import save_dataset as save_file_system_dataset

# Set DEBUG logging for unittests

log_level = logging.WARNING

logger = logging.getLogger("arcana")
logger.setLevel(log_level)

sch = logging.StreamHandler()
sch.setLevel(log_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sch.setFormatter(formatter)
logger.addHandler(sch)


@pytest.fixture(scope="session")
def nifti_sample_dir(pkg_dir):
    return pkg_dir / "test-data" / "nifti"


@pytest.fixture(scope="session")
def run_prefix():
    "A datetime string used to avoid stale data left over from previous tests"
    return datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")


@pytest.fixture
def cli_runner(catch_cli_exceptions):
    def invoke(*args, catch_exceptions=catch_cli_exceptions, **kwargs):
        runner = CliRunner()
        result = runner.invoke(*args, catch_exceptions=catch_exceptions, **kwargs)
        return result

    return invoke


@pytest.fixture
def work_dir():
    work_dir = tempfile.mkdtemp()
    return Path(work_dir)


# -----------------------
# Test dataset structures
# -----------------------


TEST_XNAT_DATASET_BLUEPRINTS = {
    "basic": TestXnatDatasetBlueprint(  # dataset name
        [1, 1, 3],  # number of timepoints, groups and members respectively
        [
            ScanBlueprint(
                "scan1",  # scan type (ID is index)
                [
                    ResourceBlueprint(
                        "Text", Text, ["file.txt"]  # resource name  # Data datatype
                    )
                ],
            ),  # name files to place within resource
            ScanBlueprint(
                "scan2",
                [ResourceBlueprint("NiftiGzX", NiftiGzX, ["file.nii.gz", "file.json"])],
            ),
            ScanBlueprint(
                "scan3",
                [
                    ResourceBlueprint(
                        "Directory", Directory, ["doubledir", "dir", "file.dat"]
                    )
                ],
            ),
            ScanBlueprint(
                "scan4",
                [
                    ResourceBlueprint(
                        "DICOM", Dicom, ["file1.dcm", "file2.dcm", "file3.dcm"]
                    ),
                    ResourceBlueprint("NIFTI", NiftiGz, ["file1.nii.gz"]),
                    ResourceBlueprint("BIDS", None, ["file1.json"]),
                    ResourceBlueprint("SNAPSHOT", None, ["file1.png"]),
                ],
            ),
        ],
        [],
        [
            DerivBlueprint("deriv1", Clinical.timepoint, Text, ["file.txt"]),
            DerivBlueprint(
                "deriv2", Clinical.subject, NiftiGzX, ["file.nii.gz", "file.json"]
            ),
            DerivBlueprint("deriv3", Clinical.batch, Directory, ["dir"]),
            DerivBlueprint("deriv4", Clinical.dataset, Text, ["file.txt"]),
        ],
    ),  # id_inference dict
    "multi": TestXnatDatasetBlueprint(  # dataset name
        [2, 2, 2],  # number of timepoints, groups and members respectively
        [ScanBlueprint("scan1", [ResourceBlueprint("Text", Text, ["file.txt"])])],
        [
            ("subject", r"group(?P<group>\d+)member(?P<member>\d+)"),
            ("session", r"timepoint(?P<timepoint>\d+).*"),
        ],  # id_inference dict
        [
            DerivBlueprint("deriv1", Clinical.session, Text, ["file.txt"]),
            DerivBlueprint(
                "deriv2", Clinical.subject, NiftiGzX, ["file.nii.gz", "file.json"]
            ),
            DerivBlueprint("deriv3", Clinical.timepoint, Directory, ["doubledir"]),
            DerivBlueprint("deriv4", Clinical.member, Text, ["file.txt"]),
            DerivBlueprint("deriv5", Clinical.dataset, Text, ["file.txt"]),
            DerivBlueprint("deriv6", Clinical.batch, Text, ["file.txt"]),
            DerivBlueprint("deriv7", Clinical.matchedpoint, Text, ["file.txt"]),
            DerivBlueprint("deriv8", Clinical.group, Text, ["file.txt"]),
        ],
    ),
    "concatenate_test": TestXnatDatasetBlueprint(
        [1, 1, 2],
        [
            ScanBlueprint("scan1", [ResourceBlueprint("Text", Text, ["file1.txt"])]),
            ScanBlueprint("scan2", [ResourceBlueprint("Text", Text, ["file2.txt"])]),
        ],
        {},
        [DerivBlueprint("concatenated", Clinical.session, Text, ["concatenated.txt"])],
    ),
}

GOOD_DATASETS = ["basic.api", "multi.api", "basic.cs", "multi.cs"]
MUTABLE_DATASETS = ["basic.api", "multi.api", "basic.cs", "multi.cs"]

# ------------------------------------
# Pytest fixtures and helper functions
# ------------------------------------


@pytest.fixture(params=GOOD_DATASETS, scope="session")
def xnat_dataset(xnat_repository, xnat_archive_dir, request):
    dataset_id, access_method = request.param.split(".")
    blueprint = TEST_XNAT_DATASET_BLUEPRINTS[dataset_id]
    run_prefix = xnat_repository.__annotations__["run_prefix"]
    with xnat4tests.connect() as login:
        if make_project_id(dataset_id, run_prefix) not in login.projects:
            create_dataset_data_in_repo(dataset_id, blueprint, run_prefix)
    return access_dataset(
        dataset_id=dataset_id,
        blueprint=blueprint,
        xnat_repository=xnat_repository,
        xnat_archive_dir=xnat_archive_dir,
        access_method=access_method,
    )


@pytest.fixture(params=MUTABLE_DATASETS, scope="function")
def mutable_xnat_dataset(xnat_repository, xnat_archive_dir, request):
    dataset_id, access_method = request.param.split(".")
    blueprint = TEST_XNAT_DATASET_BLUEPRINTS[dataset_id]
    return make_mutable_dataset(
        dataset_id=dataset_id,
        blueprint=blueprint,
        xnat_repository=xnat_repository,
        xnat_archive_dir=xnat_archive_dir,
        access_method=access_method,
        dataset_name="test",
    )


multi_store = ["file_system", "xnat"]


@pytest.fixture(params=multi_store)
def saved_dataset_multi_store(xnat_archive_dir, xnat_repository, work_dir, request):
    if request.param == "file_system":
        return save_file_system_dataset(work_dir)
    elif request.param == "xnat":
        blueprint = TestXnatDatasetBlueprint(
            dim_lengths=[1, 1, 1, 1],
            scans=["file1.txt", "file2.txt"],
            id_inference={},
            derivatives=[],
        )
        dataset = make_mutable_dataset(
            "saved_dataset",
            blueprint,
            xnat_repository,
            xnat_archive_dir,
            access_method="api",
        )
        dataset.save()
        return dataset
    else:
        assert False


@pytest.fixture(scope="session")
def xnat4tests_config() -> xnat4tests.Config:

    return xnat4tests.Config()


@pytest.fixture(scope="session")
def xnat_root_dir(xnat4tests_config) -> Path:
    return xnat4tests_config.xnat_root_dir


@pytest.fixture(scope="session")
def xnat_archive_dir(xnat_root_dir):
    return xnat_root_dir / "archive"


@pytest.fixture(scope="session")
def xnat_repository(run_prefix, xnat4tests_config):

    xnat4tests.start_xnat()

    repository = Xnat(
        server=xnat4tests_config.xnat_uri,
        user=xnat4tests_config.xnat_user,
        password=xnat4tests_config.xnat_password,
        cache_dir=mkdtemp(),
    )

    # Stash a project prefix in the repository object
    repository.__annotations__["run_prefix"] = run_prefix

    yield repository


@pytest.fixture(scope="session")
def xnat_respository_uri(xnat_repository):
    return xnat_repository.server


@pytest.fixture(scope="session")
def docker_registry_for_xnat():
    return xnat4tests.start_registry()


@pytest.fixture(scope="session")
def docker_registry_for_xnat_uri(docker_registry_for_xnat):
    if sys.platform == "linux":
        uri = "172.17.0.1"  # Linux + GH Actions
    else:
        uri = "host.docker.internal"  # Mac/Windows local debug
    return uri


@pytest.fixture
def dummy_niftix(work_dir):

    nifti_path = work_dir / "t1w.nii"
    json_path = work_dir / "t1w.json"

    # Create a random Nifti file to satisfy BIDS parsers
    hdr = nibabel.Nifti1Header()
    hdr.set_data_shape((10, 10, 10))
    hdr.set_zooms((1.0, 1.0, 1.0))  # set voxel size
    hdr.set_xyzt_units(2)  # millimeters
    hdr.set_qform(numpy.diag([1, 2, 3, 1]))
    nibabel.save(
        nibabel.Nifti1Image(
            numpy.random.randint(0, 1, size=[10, 10, 10]),
            hdr.get_best_affine(),
            header=hdr,
        ),
        nifti_path,
    )

    with open(json_path, "w") as f:
        json.dump({"test": "json-file"}, f)

    return NiftiX.from_fs_paths(nifti_path, json_path)


# For debugging in IDE's don't catch raised exceptions and let the IDE
# break at it
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value

    CATCH_CLI_EXCEPTIONS = False
else:
    CATCH_CLI_EXCEPTIONS = True


@pytest.fixture
def catch_cli_exceptions():
    return CATCH_CLI_EXCEPTIONS


@pytest.fixture(scope="session")
def command_spec():
    return {
        "task": "arcana.core.utils.testing.tasks:concatenate",
        "inputs": {
            "first_file": {
                "datatype": "common:Text",
                "field": "in_file1",
                "default_column": {
                    "row_frequency": "session",
                },
                "help_string": "the first file to pass as an input",
            },
            "second_file": {
                "datatype": "common:Text",
                "field": "in_file2",
                "default_column": {
                    "row_frequency": "session",
                },
                "help_string": "the second file to pass as an input",
            },
        },
        "outputs": {
            "concatenated": {
                "datatype": "common:Text",
                "field": "out_file",
                "help_string": "an output file",
            }
        },
        "parameters": {
            "number_of_duplicates": {
                "field": "duplicates",
                "default": 2,
                "datatype": "int",
                "required": True,
                "help_string": "a parameter",
            }
        },
        "row_frequency": "session",
    }
