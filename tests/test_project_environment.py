"""Environment gates are filesystem validation only; no dependencies installed."""
from pathlib import Path
import pytest
from scripts.check_project_environment import validate_environment


def environment(tmp_path):
    prefix=tmp_path/'.venv';prefix.mkdir()
    (prefix/'pyvenv.cfg').write_text('include-system-site-packages = false\n','utf8')
    return prefix


def test_project_owned_venv_passes(tmp_path):
    prefix=environment(tmp_path)
    result=validate_environment(tmp_path,prefix,tmp_path/'platform',(3,12))
    assert result['passed'] and not result['system_site_packages']


@pytest.mark.parametrize('case',['shared_prefix','base_python'])
def test_other_or_base_environment_is_refused(tmp_path,case):
    prefix=environment(tmp_path)
    if case=='shared_prefix':
        with pytest.raises(RuntimeError,match='outside'):validate_environment(tmp_path,tmp_path/'another-project/.venv',tmp_path/'platform',(3,12))
    else:
        with pytest.raises(RuntimeError,match='base/system'):validate_environment(tmp_path,prefix,prefix,(3,12))


def test_later_true_site_packages_setting_cannot_bypass_gate(tmp_path):
    prefix=environment(tmp_path)
    (prefix/'pyvenv.cfg').write_text('include-system-site-packages = false\ninclude-system-site-packages = true\n','utf8')
    with pytest.raises(RuntimeError,match='disable system'):validate_environment(tmp_path,prefix,tmp_path/'platform',(3,12))


@pytest.mark.parametrize('version',[(3,11),(3,13)])
def test_wrong_python_refused(tmp_path,version):
    prefix=environment(tmp_path)
    with pytest.raises(RuntimeError,match='Python 3.12'):validate_environment(tmp_path,prefix,tmp_path/'platform',version)


def test_missing_environment_metadata_is_refused(tmp_path):
    prefix=tmp_path/'.venv';prefix.mkdir()
    with pytest.raises(RuntimeError,match='pyvenv.cfg'):validate_environment(tmp_path,prefix,tmp_path/'platform',(3,12))
