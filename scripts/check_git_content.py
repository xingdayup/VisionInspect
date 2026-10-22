"""Reject private assets, credentials and large files in the Git index."""
import pathlib
import re
import subprocess


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z']).decode('utf8').split('\0')
    forbidden = {'data', 'weights', 'outputs', 'logs', '.venv', 'node_modules', '.cache', '.tmp'}
    patterns = [r'sk-(?:proj-)?[A-Za-z0-9_-]{24,}', r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    rejected = []
    total = 0
    for name in filter(None, names):
        path = pathlib.Path(name)
        if not path.is_file():
            continue
        total += path.stat().st_size
        reasons = []
        if set(pathlib.PurePosixPath(name).parts) & forbidden:
            reasons.append('private asset directory')
        if path.stat().st_size > 2 * 1024**2:
            reasons.append('larger than 2 MiB')
        if path.name.startswith('.env') and path.name != '.env.example':
            reasons.append('environment secrets')
        try:
            content = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            content = ''
        if any(re.search(pattern, content) for pattern in patterns):
            reasons.append('possible credential')
        if reasons:
            rejected.append({'path': name, 'reasons': reasons})
    print({'tracked_files': len(list(filter(None, names))), 'bytes': total, 'rejected': rejected})
    return bool(rejected)


if __name__ == '__main__':
    raise SystemExit(main())
