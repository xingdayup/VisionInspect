import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from visioninspect.api import app
from visioninspect.schemas import DetectionResult,FinalResult

client=TestClient(app)


def test_health_is_explicit_about_scope():
    response=client.get('/health')
    assert response.status_code==200
    assert response.json()['inference_service'] is response.json()['cuda_available']
    assert response.json()['device']=='CUDA'


def test_task_endpoint_never_fakes_a_result():
    response=client.post('/api/tasks',json={})
    assert response.status_code==422


def test_metrics_errors_and_schema_inventory():
    assert client.get('/api/runs/no-such-run/metrics').status_code==400
    assert client.get('/api/runs/20261021-000000-evaluate-abcdef/metrics').status_code==404
    schemas=client.get('/api/tools/schemas').json()
    assert len(schemas)==6
    assert 'review_evidence' in schemas
    assert all(value['service_implemented'] for value in schemas.values())


def test_contracts_reject_invalid_outputs():
    with pytest.raises(ValidationError):
        DetectionResult(score=float('nan'),anomaly_map_path='x',model_version='v1',elapsed_seconds=1)
    with pytest.raises(ValidationError):
        FinalResult(state='released',evidence_paths=[],rule_version='v1')
