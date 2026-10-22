"""Windows PowerShell 5.1 reads BOM-less UTF-8 as the legacy system code page."""
import codecs
import json
import os
import subprocess
from pathlib import Path

import pytest
from visioninspect.common import ROOT

SCRIPTS = [*sorted((ROOT/'scripts').glob('*.ps1')), ROOT/'docs/audit/phase0_readonly.ps1']

@pytest.mark.parametrize('script', SCRIPTS, ids=lambda path:path.name)
def test_powershell_scripts_have_explicit_utf8_bom(script):
    raw = script.read_bytes()
    assert raw.startswith(codecs.BOM_UTF8), f'Windows PowerShell 5.1 requires an explicit UTF-8 BOM: {script}'
    assert '\ufffd' not in raw.decode('utf-8-sig')

@pytest.mark.skipif(os.name!='nt', reason='Windows PowerShell 5.1 is Windows-only')
def test_windows_powershell_parses_actual_files_and_preserves_chinese_literals():
    executable = Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    if not executable.is_file():pytest.skip('Windows PowerShell not installed')
    # Command is ASCII; actual .ps1 files must be decoded by PowerShell itself.
    command = r'''
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$files = @(Get-ChildItem -LiteralPath 'scripts' -Filter '*.ps1') + @(Get-Item -LiteralPath 'docs/audit/phase0_readonly.ps1')
$rows = foreach ($file in $files) {
    $tokens = $null; $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($file.FullName,[ref]$tokens,[ref]$errors)
    [pscustomobject]@{path=$file.Name;errors=@($errors | ForEach-Object { $_.ErrorId });strings=@($tokens | Where-Object { $_.Kind -eq 'StringLiteral' } | ForEach-Object { $_.Value })}
}
[pscustomobject]@{version=$PSVersionTable.PSVersion.ToString();scripts=@($rows)} | ConvertTo-Json -Depth 6
'''
    result = subprocess.run([str(executable),'-NoProfile','-NonInteractive','-Command',command],
                            cwd=ROOT, capture_output=True,timeout=20)
    assert result.returncode==0, result.stderr.decode('utf8',errors='replace')
    report = json.loads(result.stdout.decode('utf-8-sig'))
    assert report['version'].startswith('5.1.'), report['version']
    assert len(report['scripts'])==len(SCRIPTS)
    assert all(not row['errors'] for row in report['scripts']), report
    startup = next(row for row in report['scripts'] if row['path']=='start_local.ps1')
    assert any('前端尚未构建' in value for value in startup['strings'])
