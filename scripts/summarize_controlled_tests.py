"""Extract named controlled-case rates from actual JUnit results, not visual metrics."""
import sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visioninspect.common import write_json
root=ET.parse(ROOT/'logs/completion-tests.xml').getroot();result={}
for name,marker in [('routing_oracle','routing_oracle_case'),('controlled_queue_recovery','controlled_recovery_case')]:
    cases=[]
    for case in root.iter('testcase'):
        properties={entry.get('name'):entry.get('value') for entry in case.findall('./properties/property')}
        if marker not in properties:continue
        passed=not any(case.find(tag) is not None for tag in ('failure','error','skipped'))
        cases.append(dict(name=case.get('name'),passed=passed,seconds=float(case.get('time','0'))))
    if not cases:raise ValueError('Expected controlled cases absent from JUnit')
    result[name]=dict(cases=cases,total=len(cases),passed=sum(case['passed'] for case in cases),
                      rate=sum(case['passed'] for case in cases)/len(cases))
result['scope']='Controlled unit/fault-injection cases only. Not dataset accuracy, real-world recovery probability or industrial reliability.'
write_json(ROOT/'logs/controlled-agent-metrics.json',result)
print({key:{k:v for k,v in value.items() if k!='cases'} for key,value in result.items() if isinstance(value,dict)})
