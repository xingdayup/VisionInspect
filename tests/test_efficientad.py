"""Contract/provenance tests. No synthetic model output is reported as a dataset metric."""
import pytest
from visioninspect.common import identity,write_json
from visioninspect.efficientad import authorized_regularizer

def manifest(tmp_path,**changes):
    body=dict(schema_version=1,root=str(tmp_path),records=[],source_url='https://example.invalid/unit-fixture',
              license='UNIT TEST ONLY',authorized_for_research=False,authorization_note='unit fixture, not a grant of rights')
    body.update(changes);body['digest']=identity(body);path=tmp_path/'regularizer.json';write_json(path,body);return path

def test_regularizer_requires_explicit_rights_record_and_files(tmp_path):
    with pytest.raises(ValueError,match='rights record'):authorized_regularizer(manifest(tmp_path),set())
    with pytest.raises(ValueError,match='missing'):authorized_regularizer(manifest(tmp_path,authorized_for_research=True),set())

def test_regularizer_cannot_reuse_mvtec_or_escape_root(tmp_path):
    from visioninspect.common import sha256
    path=tmp_path/'unit-fixture.bin';path.write_bytes(b'unit fixture, never an image in an experiment')
    checksum=sha256(path);records=[dict(path=path.name,sha256=checksum)]
    value=manifest(tmp_path,authorized_for_research=True,records=records)
    with pytest.raises(ValueError,match='MVTec'):authorized_regularizer(value,{checksum})
    value=manifest(tmp_path,authorized_for_research=True,records=[dict(path='../outside.jpg',sha256='x')])
    with pytest.raises(ValueError,match='unsafe'):authorized_regularizer(value,set())

def test_excessive_training_budget_is_rejected_before_training():
    from visioninspect.efficientad import enforce_fit_budget
    config=dict(runtime=dict(threads=4,fit_timeout_seconds=3600),training=dict(steps=70000))
    record=dict(cpu_threads=4,config_digest=identity(config),step_p95_seconds=2.0)
    with pytest.raises(RuntimeError,match='before training'):enforce_fit_budget(config,record)
    record['cpu_threads']=8
    with pytest.raises(ValueError,match='frozen config'):enforce_fit_budget(config,record)
