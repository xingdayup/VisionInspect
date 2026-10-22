"""Deterministic GPU detection and evidence preparation for the review graph."""
from __future__ import annotations
import time
from datetime import datetime,timezone
from pydantic import ValidationError
from .schemas import FinalResult
from .tools import ToolFailure

class InspectionAgent:
    def __init__(self,tools,policy,progress=None):
        self.tools=tools;self.policy=policy;self.progress=progress or (lambda value:None)
        self.trace=[];self.errors=[];self.started=0;self.initialization=None

    def prepare_detection(self):
        prepare=getattr(self.tools,'prepare_detector',None)
        if not callable(prepare) or len(self.trace)>=self.policy['max_tool_calls']-1:return True
        limit=min(self.policy.get('model_initialization_timeout_seconds',90),self.policy['timeout_seconds'])
        self.initialization=dict(status='running',seconds=0.,limit_seconds=limit)
        self.progress(dict(stage='initialize_model',trace=self.trace,initialization=self.initialization))
        start=time.perf_counter();failure=None
        try:
            timings=prepare()
            if time.perf_counter()-start>limit:
                raise ToolFailure('timeout','模型初始化超过时限；本次未执行异常检测')
            if time.perf_counter()-self.started>self.policy['timeout_seconds']:
                raise ToolFailure('timeout','模型初始化超出任务总时限；本次未执行异常检测')
            self.initialization.update(status='completed',timings=timings)
        except ToolFailure as error:
            failure=dict(code=error.code,message=error.message,retryable=False)
        except Exception as error:
            failure=dict(code='initialization_failed',message=f'{type(error).__name__}: {error}',retryable=False)
        self.initialization['seconds']=time.perf_counter()-start
        if failure:
            self.initialization.update(status='failed',error=failure)
            self.errors.append(f"{failure['code']}: {failure['message']}")
        self.progress(dict(stage='model_ready' if not failure else 'initialization_failed',trace=self.trace,
                           initialization=self.initialization))
        return failure is None

    def call(self,name,arguments,reason,final=False):
        attempts=self.policy['max_retries']+1
        for attempt in range(attempts):
            reserve=0 if final else 1
            if len(self.trace)>=self.policy['max_tool_calls']-reserve:
                self.errors.append('budget_exceeded: 工具调用预算已用尽');return None
            if time.perf_counter()-self.started>self.policy['timeout_seconds']:
                self.errors.append('timeout: 任务超时');return None
            item=dict(index=len(self.trace)+1,tool=name,reason=reason,input=arguments,
                      started_at=datetime.now(timezone.utc).isoformat(),attempt=attempt+1,status='running')
            self.trace.append(item);self.progress(dict(stage=name,trace=self.trace))
            start=time.perf_counter();failure=None
            try:
                result=self.tools.invoke(name,arguments)
                if time.perf_counter()-start>self.policy['tool_timeout_seconds']:
                    raise ToolFailure('timeout','工具超过耗时上限，结果作废')
                item.update(status='completed',output=result)
            except ToolFailure as error:
                failure=dict(code=error.code,message=error.message,retryable=error.retryable)
            except ValidationError as error:
                failure=dict(code='schema_error',message=str(error),retryable=False)
            except Exception as error:
                failure=dict(code='tool_failure',message=f'{type(error).__name__}: {error}',retryable=False)
            item['seconds']=time.perf_counter()-start
            if failure:
                item.update(status='failed',error=failure)
                self.progress(dict(stage=name,trace=self.trace))
                if failure['retryable'] and attempt+1<attempts:continue
                self.errors.append(f"{failure['code']}: {failure['message']}");return None
            self.progress(dict(stage=name,trace=self.trace));return result
        return None

    def run(self,image_path,category,mode='agent_review'):
        if mode not in {'baseline','agent_review'}:raise ValueError('Unknown inspection mode')
        self.started=time.perf_counter();request=dict(image_path=str(image_path),category=category)
        outputs={};quality=self.call('inspect_image_quality',request,'先检查图像是否可可靠解读')
        outputs['quality']=quality;quality_ok=bool(quality and quality['valid'])
        detection=None;ambiguous=False;conflict=False;evidence=[]
        if quality_ok:
            if self.prepare_detection():
                detection=self.call('detect_anomaly',request,'质量通过，执行本地异常检测模型')
            outputs['detection']=detection
        if detection:
            score=detection['score'];profile=self.tools.profile
            ambiguous=profile['review_lower']<=score<=profile['review_upper'] if mode=='agent_review' else False
            evidence=list(detection['evidence'].values())+[detection['anomaly_map_path']]
            if score>self.tools.threshold or ambiguous:
                regions=self.call('analyze_suspicious_region',dict(**request,anomaly_map_path=detection['anomaly_map_path']),
                                  '按校准像素阈值提取疑似区域')
                outputs['regions']=regions
                if regions:evidence+=regions['crop_paths']
                if score>self.tools.threshold and regions and not regions['crop_paths']:
                    conflict=True
            if ambiguous:
                references=self.call('retrieve_normal_reference',request,'不确定样本检索正常训练参考图')
                outputs['references']=references
                if references:
                    evidence+=references['reference_paths']
                    if score<=self.tools.threshold and max(references['similarities'],default=-1)<references['similarity_floor']:
                        conflict=True
        arguments=dict(score=detection['score'] if detection else None,threshold=self.tools.threshold,
                       evidence_paths=evidence,errors=list(self.errors),quality_ok=quality_ok,ambiguous=ambiguous,conflict=conflict)
        final=self.call('finalize_inspection',arguments,'按冻结阈值和失败降级规则汇总候选状态',final=True)
        if final is None:
            final=FinalResult(state='manual_review',evidence_paths=evidence,rule_version=self.policy['rule_version'],
                              reasons=self.errors or ['结果汇总失败']).model_dump()
        return dict(schema_version=1,category=category,mode=mode,result=final,outputs=outputs,trace=self.trace,
                    errors=self.errors,stop_reason='manual_review_required' if final['state']=='manual_review' else 'rules_complete',
                    seconds=time.perf_counter()-self.started,tool_calls=len(self.trace),vlm_calls=0,
                    monetary_cost=0,currency='USD',policy_version=self.policy['rule_version'],
                    profile_digest=self.tools.profile['digest'],initialization=self.initialization,
                    notice='研究原型候选结果，需要人工复核；不构成工业放行授权。')
