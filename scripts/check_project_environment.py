"""Read-only gate: installations must target this checkout's isolated .venv."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def validate_environment(root,prefix,base_prefix,version):
    root=Path(root).resolve();expected=root/'.venv';actual=Path(prefix).resolve()
    if tuple(version)!=(3,12):
        raise RuntimeError('Application environment must use Python 3.12')
    # Keep expected lexical after resolving the project: a .venv link must not
    # resolve to another environment, even when sys.prefix follows that same link.
    if actual!=expected or expected.is_symlink():
        raise RuntimeError('Refusing an environment outside this checkout .venv or a linked .venv')
    if actual==Path(base_prefix).resolve():
        raise RuntimeError('Refusing base/system Python; create an isolated project environment')
    try:content=(expected/'pyvenv.cfg').read_text(encoding='utf8')
    except OSError as error:raise RuntimeError('Project pyvenv.cfg is missing or unreadable') from error
    values={key.strip().lower():value.strip().lower() for line in content.splitlines()
            if '=' in line for key,value in [line.split('=',1)]}
    if values.get('include-system-site-packages')!='false':
        raise RuntimeError('Project environment must disable system site packages')
    return dict(passed=True,project_root=str(root),prefix=str(actual),base_prefix=str(Path(base_prefix).resolve()),
                python_version=list(version),system_site_packages=False)


if __name__=='__main__':
    print(json.dumps(validate_environment(ROOT,sys.prefix,sys.base_prefix,sys.version_info[:2]),indent=2))
