from pathlib import Path
import tempfile
import xnat4tests
import pytest
from arcana.core.data.set import Dataset
from arcana.core.cli.deploy import install_license
from arcana.core.utils.misc import show_cli_trace
from arcana.xnat import Xnat

LICENSE_CONTENTS = "test license"


@pytest.fixture(scope="module")
def test_license():
    tmp_dir = Path(tempfile.mkdtemp())
    test_license = tmp_dir / "license.txt"
    test_license.write_text(LICENSE_CONTENTS)
    return str(test_license)


def test_cli_install_dataset_license(
    simple_dataset: Dataset, test_license, arcana_home, cli_runner, tmp_path
):
    store_nickname = simple_dataset.id + "_store"
    dataset_name = "testing1234"
    license_name = "test-license"
    simple_dataset.store.save(store_nickname)
    dataset_locator = store_nickname + "//" + simple_dataset.id + "@" + dataset_name
    simple_dataset.save(dataset_name)

    result = cli_runner(
        install_license,
        [
            license_name,
            test_license,
            dataset_locator,
        ],
    )
    assert result.exit_code == 0, show_cli_trace(result)
    assert simple_dataset.get_license_file(license_name).contents == LICENSE_CONTENTS

    # Test overwriting
    new_contents = "new_contents"
    new_license = tmp_path / "new-license.txt"
    new_license.write_text(new_contents)

    result = cli_runner(
        install_license,
        [
            license_name,
            str(new_license),
            dataset_locator,
        ],
    )
    assert result.exit_code == 0, show_cli_trace(result)
    assert simple_dataset.get_license_file(license_name).contents == new_contents


def test_cli_install_site_license(
    xnat_repository: Xnat,
    test_license: str,
    arcana_home,
    cli_runner,
    xnat4tests_config: xnat4tests.Config,
):
    store_nickname = "site_license_store"
    license_name = "test-license"
    xnat_repository.save(store_nickname)

    result = cli_runner(
        install_license,
        [
            license_name,
            test_license,
            store_nickname,
        ],
        env={
            xnat_repository.SITE_LICENSES_USER_ENV: xnat4tests_config.xnat_user,
            xnat_repository.SITE_LICENSES_PASS_ENV: xnat4tests_config.xnat_password,
        },
    )

    assert result.exit_code == 0, show_cli_trace(result)

    assert (
        xnat_repository.get_site_license_file(
            license_name,
            user=xnat4tests_config.xnat_user,
            password=xnat4tests_config.xnat_password,
        ).contents
        == LICENSE_CONTENTS
    )
