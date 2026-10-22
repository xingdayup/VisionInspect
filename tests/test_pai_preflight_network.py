"""Metadata transport regressions; responses are fixtures, not live connectivity tests."""
from http.client import IncompleteRead
from io import BytesIO
import ssl
import json
import sys
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from packaging.tags import Tag
import pytest
from scripts import preflight_pai_app as preflight


class Response:
    headers={'Content-Length':'123'}
    def __init__(self,body=b'{"urls":[]}',error=None):
        self.body=body;self.error=error;self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True
    def read(self,limit):
        if self.error:raise self.error
        return self.body[:limit]


def test_timeout_while_reading_body_retries_and_closes_response(monkeypatch):
    first=Response(error=TimeoutError('body read timed out'));second=Response()
    responses=iter([first,second]);calls=[];sleeps=[]
    def open_url(request,timeout):
        calls.append((request.full_url,timeout));return next(responses)
    monkeypatch.setattr(preflight.urllib.request,'urlopen',open_url)
    monkeypatch.setattr(preflight.time,'sleep',sleeps.append)
    assert preflight.fetch_metadata('https://pypi.org/pypi/example/1/json')==second.body
    assert len(calls)==2 and sleeps==[2] and first.closed and second.closed
    assert all(timeout==90 for _,timeout in calls)


@pytest.mark.parametrize('method',['GET','HEAD'])
def test_connection_failures_exhaust_bounded_attempts(monkeypatch,method):
    calls=[];sleeps=[]
    def open_url(request,timeout):
        calls.append(request.get_method());raise URLError(TimeoutError('connect timed out'))
    monkeypatch.setattr(preflight.urllib.request,'urlopen',open_url)
    monkeypatch.setattr(preflight.time,'sleep',sleeps.append)
    with pytest.raises(RuntimeError,match='3 attempts'):
        preflight.fetch_metadata('https://pypi.org/pypi/example/1/json',method=method)
    assert calls==[method]*3 and sleeps==[2,4]


@pytest.mark.parametrize('error',[
    HTTPError('https://pypi.org/example',404,'missing',{},None),
    URLError(ssl.SSLCertVerificationError('certificate verification failed')),
])
def test_permanent_http_or_certificate_failure_is_not_bypassed(monkeypatch,error):
    calls=[]
    def open_url(request,timeout):calls.append(request);raise error
    monkeypatch.setattr(preflight.urllib.request,'urlopen',open_url)
    with pytest.raises(type(error)):preflight.fetch_metadata('https://pypi.org/example')
    assert len(calls)==1


def test_incomplete_response_and_temporary_http_failure_retry(monkeypatch):
    attempts=iter([HTTPError('https://pypi.org/example',503,'unavailable',{},BytesIO()),
                   Response(error=IncompleteRead(b'partial')),Response()])
    def open_url(request,timeout):
        item=next(attempts)
        if isinstance(item,Exception):raise item
        return item
    monkeypatch.setattr(preflight.urllib.request,'urlopen',open_url)
    monkeypatch.setattr(preflight.time,'sleep',lambda seconds:None)
    assert preflight.fetch_metadata('https://pypi.org/example')==b'{"urls":[]}'


def test_head_size_and_metadata_size_limit(monkeypatch):
    response=Response()
    monkeypatch.setattr(preflight.urllib.request,'urlopen',lambda *args,**kwargs:response)
    assert preflight.fetch_metadata('https://download.pytorch.org/example',method='HEAD')==123
    response.body=b'x'*(8*1024**2+1)
    with pytest.raises(ValueError,match='8 MiB'):preflight.fetch_metadata('https://pypi.org/example')


@pytest.mark.parametrize('wheel_bytes,free_bytes,passed',[
    (123,16*1024**3,True),
    (1_500_000_001,16*1024**3,False),
    (123,14*1024**3,False),
])
def test_completed_transport_still_enforces_resource_budgets(tmp_path,monkeypatch,wheel_bytes,free_bytes,passed):
    (tmp_path/'requirements-agent-linux.txt').write_text('example==1.0\n','utf8')
    monkeypatch.setattr(preflight,'ROOT',tmp_path)
    monkeypatch.setattr(preflight,'sys',SimpleNamespace(platform='linux',version_info=(3,12),
        version=sys.version,executable=sys.executable))
    monkeypatch.setattr(preflight.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(preflight,'sys_tags',lambda:iter([Tag('py3','none','any')]))
    monkeypatch.setattr(preflight.shutil,'disk_usage',lambda path:SimpleNamespace(free=free_bytes))
    metadata=dict(urls=[dict(filename='example-1.0-py3-none-any.whl',size=wheel_bytes,
        url='https://files.pythonhosted.org/example-1.0-py3-none-any.whl')])
    monkeypatch.setattr(preflight,'fetch_metadata',lambda url:json.dumps(metadata).encode())
    if passed:preflight.main([])
    else:
        with pytest.raises(RuntimeError,match='resource budget exceeded'):preflight.main([])
    report=json.loads((tmp_path/'logs/pai-app-preflight.json').read_text('utf8'))
    assert report['passed'] is passed and not report['packages_installed']
