import { useEffect, useRef, useState, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import './style.css';

type View = 'workbench' | 'evidence' | 'trace' | 'history' | 'report' | 'system';
type Model = { category: string; label: string; model: string; available: boolean; reason: string };
type Health = { status: string; device: string; models: Model[]; inference_service: boolean; vlm_available: boolean; weights_present: boolean; dataset_manifest_present: boolean; worker_concurrency: number };
type Trace = { index: number; tool: string; reason: string; input: unknown; output?: unknown; status: string; seconds?: number; attempt: number; error?: { code: string; message: string } };
type Detection = { score: number; threshold: number; model_version: string; elapsed_seconds: number; anomaly_map_path: string; evidence: Record<string, string> };
type Region = { id: string; crop_path: string; pixels: number; maximum_score: number; box: number[] };
type Initialization = { status: string; seconds: number; limit_seconds: number; error?: { code: string; message: string } };
type Report = { result: { state: string; reasons: string[]; evidence_paths: string[]; rule_version?: string }; outputs: { quality?: { valid: boolean; issues: string[]; width: number; height: number }; detection?: Detection; references?: { reference_paths: string[]; similarities: number[]; source_split: string }; regions?: { regions: Region[]; anomaly_area_ratio: number }; candidate?: { available: boolean; raw_text: string; uncertain: boolean } }; trace: Trace[]; seconds: number; tool_calls: number; vlm_calls: number; monetary_cost: number; initialization?: Initialization | null; stop_reason: string; policy_version: string; profile_digest: string; checkpoint_sha256: string; manifest_digest: string; notice: string; errors: string[] };
type Review = { decision: string; note: string; recorded_at: string };
type Task = { id: string; created_at: string; updated_at: string; status: string; category: string; model: string; mode: string; filename: string; state: string | null; error: string | null; seconds: number | null; review: Review | null; progress?: { stage: string; trace: Trace[] }; report?: Report };
type History = { items: Task[]; total: number; limit: number; offset: number };
const apiBase = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
const nav: { key: View; title: string; path: string }[] = [
  { key: 'workbench', title: '质检工作台', path: 'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z' },
  { key: 'evidence', title: '视觉证据', path: 'M3 4h18v16H3zM3 15l5-5 4 4 3-3 6 6M15 8h.01' },
  { key: 'trace', title: '执行轨迹', path: 'M6 4v16M6 6h12M6 12h8M6 18h12M3 6h.01M3 12h.01M3 18h.01' },
  { key: 'history', title: '检测历史', path: 'M3 12a9 9 0 1 0 3-7M3 3v5h5M12 7v5l3 2' },
  { key: 'report', title: '质检报告', path: 'M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7' },
  { key: 'system', title: '系统状态', path: 'M3 4h18v13H3zM8 21h8M12 17v4M6 11h3l2-4 3 7 2-3h2' },
];
const labels: Record<string, string> = { normal_candidate: '正常候选', anomaly_candidate: '异常候选', manual_review: '待人工复核', queued: '排队中', running: '检测中', completed: '已完成', failed: '失败', baseline: '单模型', dynamic: '动态 Agent', fixed: '固定复核', inspect_image_quality: '图像质量检查', detect_anomaly: '异常检测', retrieve_normal_reference: '正常参考检索', analyze_suspicious_region: '疑似区域分析', classify_defect_candidate: 'VLM 候选解释', finalize_inspection: '规则汇总' };
function label(value: string | null) { return value ? labels[value] ?? value : '等待结果'; }
function seconds(value?: number | null) { return value == null ? '—' : `${value.toFixed(2)} s`; }
Object.assign(labels, { initialize_model: '正在加载本地检测模型', model_ready: '检测模型已就绪', initialization_failed: '检测模型初始化失败' });

function evidenceIssue(task: Task, pending: boolean) {
  const errors = [task.error, ...(task.report?.errors ?? [])].filter(Boolean).join('；');
  if (pending) return { title: '证据正在生成', message: '任务仍在执行，可查看实时执行轨迹。' };
  if (/timeout|超时|超过.*时限|超过耗时/.test(errors)) return { title: '检测超时', message: '模型初始化或工具执行超过时限，本次没有有效检测结果。请查看执行轨迹后重新提交。' };
  if (task.report?.initialization?.status === 'failed') return { title: '模型初始化失败', message: errors || '本地模型加载失败，请查看执行轨迹与任务日志。' };
  if (errors || task.status === 'failed') return { title: '检测未能完成', message: errors || '任务执行失败，请查看执行轨迹或重新提交。' };
  const issues = task.report?.outputs.quality?.issues.filter(Boolean).join('；');
  if (task.report?.outputs.quality?.valid === false) return { title: '图像质量未通过', message: issues || '请检查图片清晰度、曝光和分辨率后重新提交。' };
  return { title: '检测证据不可用', message: '没有可用的模型结果，请查看执行轨迹后重新提交。' };
}
function date(value: string) { return new Date(value).toLocaleString('zh-CN', { hour12: false }); }
async function request<T,>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try { response = await fetch(apiBase + path, { ...init, signal: init.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(12000)]) : AbortSignal.timeout(12000) }); }
  catch (error) { throw new Error(error instanceof DOMException && (error.name === 'TimeoutError' || error.name === 'AbortError') ? '连接超时，请重试或检查后端服务。' : '无法连接后端，请确认本地服务已启动。'); }
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try { const body = await response.json(); message = typeof body.detail === 'string' ? body.detail : message; } catch { /* Keep readable HTTP status. */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
function Icon({ path, size = 19 }: { path: string; size?: number }) { return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={path} /></svg>; }
function Badge({ value }: { value: string | null }) { return <span className={`badge ${value ?? ''}`}><span className="status-dot" />{label(value)}</span>; }
function Empty({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) { return <div className="empty"><Icon size={32} path="M4 4h16v16H4zM8 8h8M8 12h5M8 16h3" /><h2>{title}</h2><p>{children}</p>{action}</div>; }

function App() {
  const [view, setView] = useState<View>('workbench');
  const [health, setHealth] = useState<Health | null>(null);
  const [history, setHistory] = useState<History>({ items: [], total: 0, limit: 20, offset: 0 });
  const [task, setTask] = useState<Task | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [category, setCategory] = useState('bottle');
  const [mode, setMode] = useState('dynamic');
  const [vlm, setVlm] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [filter, setFilter] = useState('');
  const [query, setQuery] = useState('');
  const [offset, setOffset] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const [pollFailed, setPollFailed] = useState(false);
  const [imageMode, setImageMode] = useState('overlay');
  const [reviewDecision, setReviewDecision] = useState('manual_review');
  const [reviewNote, setReviewNote] = useState('');
  const [savingReview, setSavingReview] = useState(false);
  const [reviewMessage, setReviewMessage] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const pending = task?.status === 'queued' || task?.status === 'running';
  const report = task?.report;
  const detection = report?.outputs?.detection;
  const trace = report?.trace ?? task?.progress?.trace ?? [];
  const categories = health?.models.filter(model => model.model === 'patchcore') ?? [];
  const selectedModel = categories.find(model => model.category === category);
  const selectedTitle = nav.find(item => item.key === view)!.title;

  useEffect(() => { window.scrollTo({ top: 0, left: 0, behavior: 'instant' }); }, [view]);
  useEffect(() => { setReviewNote(''); setReviewMessage(''); setReviewDecision(task?.review?.decision ?? 'manual_review'); }, [task?.id]);

  useEffect(() => {
    const controller = new AbortController(); let active = true;
    setLoading(true);
    Promise.all([request<Health>('/health', { signal: controller.signal }), request<History>(`/api/tasks?limit=20&offset=${offset}&state=${filter}&q=${encodeURIComponent(query)}`, { signal: controller.signal })])
      .then(([state, items]) => { setHealth(state); setHistory(items); setError(''); })
      .catch(reason => { if (active) { setHealth(null); setError(String(reason.message)); } })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; controller.abort(); };
  }, [refresh, filter, query, offset]);

  useEffect(() => {
    const stored = localStorage.getItem('visioninspect-task');
    if (stored) request<Task>(`/api/tasks/${stored}`).then(setTask).catch(() => localStorage.removeItem('visioninspect-task'));
  }, []);
  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file); setPreview(url); return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (!task || !pending || pollFailed) return;
    const controller = new AbortController(); let timer: ReturnType<typeof setTimeout>; let failures = 0;
    async function poll() {
      try {
        const value = await request<Task>(`/api/tasks/${task!.id}`, { signal: controller.signal }); setTask(value); failures = 0;
        if (value.status === 'completed' || value.status === 'failed') { setRefresh(n => n + 1); setView('evidence'); return; }
      } catch (reason) { if (controller.signal.aborted) return; failures++; if (failures >= 3) { setPollFailed(true); setError(String((reason as Error).message)); return; } }
      timer = setTimeout(() => void poll(), 1000);
    }
    timer = setTimeout(() => void poll(), 500);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [task?.id, pending, pollFailed]);

  function chooseFile(value?: File) {
    if (!value) return;
    setError('');
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(value.type)) { setFile(null); setError('请选择 PNG、JPEG 或 WebP 图像。'); return; }
    if (value.size > 10 * 1024 * 1024) { setFile(null); setError('图像超过 10 MiB 上限，请选择较小的文件。'); return; }
    if (value.size === 0) { setFile(null); setError('文件为空，请重新选择。'); return; }
    setFile(value);
  }
  async function submit(event: React.FormEvent) {
    event.preventDefault(); if (!file || submitting || pending) return;
    setSubmitting(true); setError(''); setReviewMessage(''); setPollFailed(false);
    const body = new FormData(); body.append('file', file); body.append('category', category); body.append('model', 'patchcore'); body.append('mode', mode); body.append('vlm', String(mode !== 'baseline' && vlm));
    try { const value = await request<Task>('/api/tasks', { method: 'POST', body }); setTask(value); localStorage.setItem('visioninspect-task', value.id); setRefresh(n => n + 1); }
    catch (reason) { setError((reason as Error).message); }
    finally { setSubmitting(false); }
  }
  async function selectTask(id: string) {
    setError(''); setReviewMessage(''); setReviewNote(''); setPollFailed(false);
    try { const value = await request<Task>(`/api/tasks/${id}`); setTask(value); localStorage.setItem('visioninspect-task', id); setView(value.status === 'queued' || value.status === 'running' ? 'workbench' : 'evidence'); }
    catch (reason) { setError((reason as Error).message); }
  }
  async function saveReview(event: React.FormEvent) {
    event.preventDefault(); if (!task || !reviewNote.trim()) return; setSavingReview(true); setReviewMessage(''); setError('');
    try { await request(`/api/tasks/${task.id}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision: reviewDecision, note: reviewNote.trim() }) }); setTask(await request<Task>(`/api/tasks/${task.id}`)); setReviewMessage('复核记录已保存，原始模型结论保持可追溯。'); setRefresh(n => n + 1); }
    catch (reason) { setError((reason as Error).message); }
    finally { setSavingReview(false); }
  }
  function asset(path: string) { return `${apiBase}/api/tasks/${task!.id}/artifacts/${encodeURIComponent(path.split('/').pop()!)}`; }
  function image(path: string, alt: string, className = '') { return <img className={className} src={asset(path)} alt={alt} loading="lazy" onError={event => { event.currentTarget.alt = `${alt}（证据无法读取）`; event.currentTarget.classList.add('image-error'); }} />; }
  function taskTable(items: Task[], compact = false) {
    return <div className="table-scroll"><table><caption className="sr-only">真实检测任务记录</caption><thead><tr><th>文件 / 任务</th><th>候选状态</th>{!compact && <><th>工作流</th><th>耗时</th></>}<th>提交时间</th><th><span className="sr-only">打开任务</span></th></tr></thead><tbody>{items.map(item => <tr key={item.id}><td><strong>{item.filename}</strong><small>{item.id.slice(0, 12)}</small></td><td><Badge value={item.status === 'completed' ? item.state : item.status} />{item.review && <small>已有人工复核</small>}</td>{!compact && <><td>{label(item.mode)}</td><td>{seconds(item.seconds)}</td></>}<td className="muted nowrap">{date(item.created_at)}</td><td><button className="text-button" onClick={() => void selectTask(item.id)}>查看<span aria-hidden="true"> ↗</span></button></td></tr>)}</tbody></table></div>;
  }
  const noTask = <Empty title="先选择一条检测任务" action={<button onClick={() => setView('workbench')}>前往工作台</button>}>上传产品图像，或从检测历史中打开任务，即可查看证据、执行轨迹和报告。</Empty>;
  const issue = task ? evidenceIssue(task, Boolean(pending)) : null;

  return <div className="app-shell">
    <a className="skip" href="#content">跳至主要内容</a>
    <aside className="sidebar">
      <a href="#" className="brand" onClick={event => { event.preventDefault(); setView('workbench'); }}><span className="brand-mark"><Icon path="M5 3H3v5M19 3h2v5M3 16v5h5M21 16v5h-5M7 12l3 3 7-7" size={25} /></span><span>VisionInspect<small>视觉质检工作台</small></span></a>
      <div className="nav-label">工作空间</div>
      <nav aria-label="主要导航">{nav.map(item => <button key={item.key} className={view === item.key ? 'nav-item selected' : 'nav-item'} aria-current={view === item.key ? 'page' : undefined} onClick={() => setView(item.key)}><Icon path={item.path} />{item.title}{item.key === 'evidence' && task?.state === 'manual_review' && <span className="nav-count">1</span>}</button>)}</nav>
      <div className="sidebar-footer"><span className={`connection ${health ? 'online' : ''}`}><span className="status-dot" />{loading ? '正在连接' : health ? '本地服务已连接' : '后端未连接'}</span><small>CPU · 本地运行</small><p>研究原型<br />候选结果需人工复核</p></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div><span>工作空间</span><span className="slash">/</span><strong>{selectedTitle}</strong></div><div className="topbar-right"><span className="outline-tag">CPU FP32</span><button className="icon-button" title="刷新连接和记录" aria-label="刷新连接和记录" onClick={() => { setRefresh(n => n + 1); setPollFailed(false); }}><Icon path="M20 8a8 8 0 1 0 0 8M20 3v5h-5" /></button></div></header>
      <main id="content" tabIndex={-1}>
        <div className="page-heading"><div><p className="eyebrow">{view === 'workbench' ? 'INSPECTION' : 'VISIONINSPECT / LOCAL'}</p><h1>{selectedTitle}</h1><p>{view === 'workbench' ? '上传产品图像，从模型判断到证据复核。' : view === 'history' ? '追溯每次检测的输入、证据与人工记录。' : view === 'system' ? '检查本地服务与模型资产的可用性。' : task ? `${task.filename} · ${task.id.slice(0, 12)}` : '选择任务后查看完整检测记录。'}</p></div>{task && ['evidence', 'trace', 'report'].includes(view) && <button className="secondary" onClick={() => setView('history')}>切换任务</button>}</div>
        {error && <div className="alert error" role="alert"><strong>操作未完成</strong><span>{error}</span><button className="text-button" onClick={() => { setError(''); setRefresh(n => n + 1); setPollFailed(false); }}>重试连接</button></div>}
        {view === 'workbench' && <>
          <div className="workbench-grid"><section className="panel upload-panel" aria-labelledby="upload-heading"><div className="section-heading"><h2 id="upload-heading">新建检测</h2><span className="step-label">01 / 选择图像</span></div>
            <form onSubmit={event => void submit(event)}>
              <div className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`} onDragOver={event => { event.preventDefault(); if (!pending) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); if (!pending) chooseFile(event.dataTransfer.files[0]); }}>
                {preview ? <><img src={preview} alt="待检测图像预览" className="upload-preview" /><strong className="filename">{file?.name}</strong><small>{((file?.size ?? 0) / 1024).toFixed(0)} KB</small><button type="button" className="text-button" disabled={pending || submitting} onClick={() => fileInput.current?.click()}>更换图像</button></> : <><div className="upload-symbol"><Icon path="M12 16V4M7 9l5-5 5 5M4 16v4h16v-4" size={29} /></div><h3>将产品图像拖到这里</h3><p>或从本机选择一张图片</p><button type="button" className="secondary" onClick={() => fileInput.current?.click()}>选择图像</button><small>PNG、JPEG、WebP · 最大 10 MiB</small></>}
                <input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp" className="sr-only" aria-label="选择待检测图像" disabled={pending || submitting} onChange={event => chooseFile(event.target.files?.[0])} />
              </div>
              <div className="form-grid"><label>产品类别<select value={category} onChange={event => setCategory(event.target.value)} disabled={pending || submitting || !categories.length}>{categories.length ? categories.map(model => <option key={model.category} value={model.category}>{model.label} / {model.category}</option>) : <option value="bottle">正在读取类别</option>}</select></label><label>检测模型<select value="patchcore" disabled><option value="patchcore">PatchCore · ResNet18</option></select></label></div>
              <label>工作流<select value={mode} onChange={event => setMode(event.target.value)} disabled={pending || submitting}><option value="dynamic">动态 Agent · 按需复核</option><option value="baseline">单模型 · 阈值判断</option><option value="fixed">固定复核 · 参考与区域分析</option></select></label>
              <label className="checkbox-label"><input type="checkbox" checked={vlm} onChange={event => setVlm(event.target.checked)} disabled={pending || submitting || !health?.vlm_available || mode === 'baseline'} /><span>启用本地 VLM 候选解释<small>{health?.vlm_available ? '辅助描述可见差异，耗时较长，不影响模型分数。' : '本地 VLM 资产尚不可用。'}</small></span></label>
              {selectedModel && !selectedModel.available && <p className="inline-error" role="alert">{selectedModel.reason}</p>}
              <div className="submit-row"><span>仅在本地处理图像</span><button className="primary" type="submit" disabled={!file || !health?.inference_service || !selectedModel?.available || submitting || pending}>{submitting ? '正在上传…' : pending ? '任务处理中…' : '开始检测'}<Icon path="M5 12h14M14 7l5 5-5 5" size={17} /></button></div>
            </form>
          </section><aside className="workflow-notes"><p className="eyebrow">检查流程</p><h2>结论背后，<br />都有可查看的证据。</h2><ol><li><span>01</span><div><strong>质量与异常检测</strong><p>先检查输入质量，再计算图像分数和像素热图。</p></div></li><li><span>02</span><div><strong>按需获取复核证据</strong><p>接近阈值时检索正常参考图，提取高分区域。</p></div></li><li><span>03</span><div><strong>形成候选结果</strong><p>规则汇总结果，证据不足时转入人工复核。</p></div></li></ol><div className="note"><Icon path="M12 3l9 17H3zM12 9v4M12 17h.01" /><p>候选状态用于研究复核，不构成工业产线的放行授权。</p></div></aside></div>
          {pending && task && <section className="panel progress-panel" aria-live="polite"><div className="section-heading"><h2>{label(task.status)} · {task.filename}</h2><span>{task.id.slice(0, 12)}</span></div><progress max={6} value={Math.min(trace.filter(item => item.status === 'completed').length, 5)} /><p>{label(task.progress?.stage ?? task.status)}{pollFailed && ' · 轮询暂停，请重试连接'}</p><button className="text-button" onClick={() => setView('trace')}>查看实时执行轨迹 ↗</button></section>}
          <section className="history-section"><div className="section-heading"><h2>最近检测</h2><button className="text-button" onClick={() => { setFilter(''); setQuery(''); setOffset(0); setView('history'); }}>全部记录 ↗</button></div>{loading ? <div className="skeleton" aria-label="正在读取记录" /> : history.items.length ? taskTable(history.items.slice(0, 5), true) : <Empty title="开始第一条检测">选择图像并提交，完成后的记录会保存在这里。</Empty>}</section>
        </>}
        {view === 'evidence' && (task ? <>
          <section className="result-strip"><div><Badge value={task.status === 'completed' ? task.state : task.status} /><p>{report?.result.reasons.join('；') ?? task.error ?? '任务正在运行，证据将在完成后显示。'}</p>{detection && <p className="caption">模型 {task.model} · {detection.model_version} · 规则 {report?.result.rule_version ?? report?.policy_version ?? '未记录'}</p>}</div><dl><div><dt>异常分数</dt><dd>{detection?.score.toFixed(4) ?? '—'}</dd></div><div><dt>冻结阈值</dt><dd>{detection?.threshold.toFixed(4) ?? '—'}</dd></div><div><dt>模型推理</dt><dd>{seconds(detection?.elapsed_seconds)}</dd></div></dl></section>
          {detection ? <><div className="evidence-toolbar"><h2>图像对照</h2><ToggleGroup.Root className="segmented" type="single" value={imageMode} onValueChange={value => { if (value) setImageMode(value); }} orientation="horizontal" aria-label="右侧证据类型"><ToggleGroup.Item value="overlay" aria-label="热图叠加">热图叠加</ToggleGroup.Item><ToggleGroup.Item value="heatmap" aria-label="异常热图">异常热图</ToggleGroup.Item></ToggleGroup.Root></div><div className="image-comparison"><figure>{image(detection.evidence.original, '检测输入原图')}<figcaption><strong>原图</strong><span>{report?.outputs.quality?.width} × {report?.outputs.quality?.height}</span></figcaption></figure><figure>{image(detection.evidence[imageMode], imageMode === 'overlay' ? '异常热图与原图叠加' : '原始异常分数的可视化热图')}<figcaption><strong>{imageMode === 'overlay' ? '热图叠加' : '异常热图'}</strong><span>256 × 256</span></figcaption></figure></div><p className="caption">亮色表示本图内相对高分，不等于已确认缺陷。请结合分数、冻结阈值和原图复核。<a href={asset(detection.anomaly_map_path)} download>下载原始异常图 .npy</a></p>
            <section className="evidence-section"><div className="section-heading"><h2>疑似区域</h2><span>{report?.outputs.regions ? `高分像素占比 ${(report.outputs.regions.anomaly_area_ratio * 100).toFixed(2)}%` : '本次未触发区域工具'}</span></div>{report?.outputs.regions?.regions.length ? <div className="region-list">{report.outputs.regions.regions.map(region => <figure key={region.id}>{image(region.crop_path, `${region.id} 疑似区域裁剪`)}<figcaption><strong>{region.id}</strong><span>{region.pixels} 个像素 · 最高分 {region.maximum_score.toFixed(3)}</span><small>原图框 [{region.box.join(', ')}]</small></figcaption></figure>)}</div> : <p className="muted">{report?.outputs.regions ? '冻结像素阈值下未形成满足大小限制的连通区域。' : '当前工作流根据模型分数决定是否提取区域。'}</p>}</section>
            <section className="evidence-section"><div className="section-heading"><h2>正常参考图</h2><span>仅来自正常训练划分</span></div>{report?.outputs.references ? <div className="reference-list">{report.outputs.references.reference_paths.map((path, i) => <figure key={path}>{image(path, `正常训练参考图 ${i + 1}`)}<figcaption><strong>参考 {i + 1}</strong><span>特征相似度 {report.outputs.references!.similarities[i].toFixed(5)}</span></figcaption></figure>)}</div> : <p className="muted">本次未触发参考检索。动态 Agent 仅对不确定样本调用此工具。</p>}</section>
            {report?.outputs.candidate && <section className="evidence-section"><h2>候选解释 <span className="outline-tag">未经验证</span></h2><p>{report.outputs.candidate.available ? report.outputs.candidate.raw_text || '模型未返回可读描述。' : '本次本地 VLM 不可用。'}</p><p className="caption">候选解释不参与产品放行，不作为缺陷分类的已验证标签。</p></section>}
          </> : <Empty title={issue!.title} action={<button className="secondary" onClick={() => setView('trace')}>查看执行轨迹</button>}>{issue!.message}</Empty>}
          {!pending && <section className="panel review-panel"><div className="section-heading"><h2>人工复核记录</h2><span>保留原始模型结论</span></div>{task.review && <div className="saved-review"><Badge value={task.review.decision} /><p>{task.review.note}</p><small>{date(task.review.recorded_at)}</small></div>}<form onSubmit={event => void saveReview(event)}><div className="form-grid review-grid"><label>复核候选状态<select value={reviewDecision} onChange={event => setReviewDecision(event.target.value)}><option value="manual_review">继续人工复核</option><option value="normal_candidate">正常候选</option><option value="anomaly_candidate">异常候选</option></select></label><label>复核说明<textarea rows={2} maxLength={2000} required value={reviewNote} onChange={event => setReviewNote(event.target.value)} placeholder="记录你观察到的证据或需要进一步确认的内容" /></label></div><div className="submit-row"><p role="status">{reviewMessage || '此记录用于研究复核，不代表产线放行。'}</p><button type="submit" className="secondary" disabled={savingReview || !reviewNote.trim()}>{savingReview ? '保存中…' : '保存复核'}</button></div></form></section>}
        </> : noTask)}
        {view === 'trace' && (task ? <><div className="trace-summary"><Badge value={task.status} /><span>{trace.length} 次工具调用</span><span>端到端 {seconds(report?.seconds)}</span><span>费用 $0 · 本地执行</span></div>{trace.length ? <ol className="trace-list">{trace.map(item => <li key={item.index}><details open={item.status === 'failed' || item.status === 'running'}><summary><span className="trace-number">{String(item.index).padStart(2, '0')}</span><span className="trace-title"><strong>{label(item.tool)}</strong><small>{item.reason}</small></span><Badge value={item.status} /><span className="trace-duration">{seconds(item.seconds)}</span></summary><div className="trace-detail"><p><strong>调用原因：</strong>{item.reason}</p><p>第 {item.attempt} 次尝试{item.error && ` · ${item.error.code}: ${item.error.message}`}</p><div className="trace-io"><div><h3>输入</h3><pre>{JSON.stringify(item.input, null, 2)}</pre></div><div><h3>输出</h3><pre>{JSON.stringify(item.output ?? item.error ?? '工具执行中…', null, 2)}</pre></div></div></div></details></li>)}</ol> : <Empty title="任务等待执行">任务已进入本地队列，开始运行后会逐步显示真实调用记录。</Empty>}<p className="caption">上限 8 次工具调用、单任务 180 秒、可重试工具最多重试 1 次。工具失败或证据冲突将转为人工复核。</p></> : noTask)}
        {view === 'trace' && report?.initialization && <p className="caption">本任务模型初始化 {seconds(report.initialization.seconds)} · {label(report.initialization.status)}。初始化与下方工具执行分别计时。</p>}
        {view === 'history' && <><div className="history-toolbar"><label className="search-label"><span className="sr-only">搜索文件名或任务编号</span><Icon path="M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0" /><input type="search" placeholder="搜索文件名或任务编号" value={query} onChange={event => { setQuery(event.target.value); setOffset(0); }} /></label><label className="filter-label"><span className="sr-only">按候选状态筛选</span><select value={filter} onChange={event => { setFilter(event.target.value); setOffset(0); }}><option value="">全部候选状态</option><option value="manual_review">待人工复核</option><option value="anomaly_candidate">异常候选</option><option value="normal_candidate">正常候选</option></select></label><span className="muted">共 {history.total} 条</span></div>{loading ? <div className="skeleton" /> : history.items.length ? taskTable(history.items) : <Empty title="没有匹配的检测记录">调整筛选条件，或前往工作台上传图像。</Empty>}<div className="pagination"><button className="secondary" disabled={!offset || loading} onClick={() => setOffset(n => Math.max(0, n - 20))}>上一页</button><span>第 {Math.floor(offset / 20) + 1} 页</span><button className="secondary" disabled={offset + 20 >= history.total || loading} onClick={() => setOffset(n => n + 20)}>下一页</button></div></>}
        {view === 'report' && (task ? report ? <><div className="report-heading"><div><Badge value={report.result.state} /><h2>{task.filename}</h2><p>{report.notice}</p></div><a className="button primary" href={`${apiBase}/api/tasks/${task.id}/report`} download={`inspection-${task.id}.json`}><Icon path="M12 3v12M7 10l5 5 5-5M4 17v4h16v-4" />导出 JSON 报告</a></div><section className="report-section"><h2>判断与依据</h2><ul>{report.result.reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul><dl className="facts"><div><dt>图像异常分数</dt><dd>{detection?.score.toFixed(6) ?? '未生成'}</dd></div><div><dt>图像阈值</dt><dd>{detection?.threshold.toFixed(6) ?? '未生成'}</dd></div><div><dt>工作流</dt><dd>{label(task.mode)}</dd></div><div><dt>工具调用 / VLM 调用</dt><dd>{report.tool_calls ?? '—'} / {report.vlm_calls ?? '—'}</dd></div><div><dt>任务总耗时</dt><dd>{seconds(task.seconds)}</dd></div><div><dt>提交时间</dt><dd>{date(task.created_at)}</dd></div></dl></section><section className="report-section"><h2>人工复核</h2>{task.review ? <><Badge value={task.review.decision} /><p>{task.review.note}</p><small>{date(task.review.recorded_at)}</small></> : <p className="muted">尚未保存人工复核记录。可在视觉证据页记录复核依据。</p>}</section><details className="provenance"><summary>可复现信息与完整报告</summary><dl className="facts"><div><dt>模型 SHA256</dt><dd className="hash">{report.checkpoint_sha256 ?? '未加载'}</dd></div><div><dt>数据清单</dt><dd className="hash">{report.manifest_digest ?? '未加载'}</dd></div><div><dt>规则 / 策略</dt><dd>{report.result.rule_version ?? report.policy_version}<small className="hash">{report.profile_digest}</small></dd></div></dl><pre>{JSON.stringify(report, null, 2)}</pre></details></> : <Empty title={pending ? '报告尚未生成' : '报告文件不可用'}>{pending ? '等待当前任务完成后，可以查看并导出报告。' : '任务已经结束，但报告文件缺失。请检查任务日志或重新提交图像。'}</Empty> : noTask)}
        {view === 'system' && <><section className="system-status"><div className={`connection ${health ? 'online' : ''}`}><span className="status-dot" />{health ? '本地服务在线' : loading ? '正在检查连接' : '后端不可用'}</div><p>默认仅本机访问，所有检测使用 CPU。系统不会自动下载数据或权重。</p><button className="secondary" onClick={() => setRefresh(n => n + 1)}>重新检查</button></section><section className="report-section"><h2>运行环境</h2><dl className="facts"><div><dt>推理设备 / 精度</dt><dd>{health ? 'CPU / FP32' : '等待连接'}</dd></div><div><dt>任务并发</dt><dd>{health?.worker_concurrency ?? '—'}</dd></div><div><dt>付费 API</dt><dd>未接入</dd></div><div><dt>本地 VLM</dt><dd>{health?.vlm_available ? '文件已准备，可选启用' : '不可用'}</dd></div><div><dt>骨干权重</dt><dd>{health?.weights_present ? '文件存在' : '不可用'}</dd></div><div><dt>数据清单</dt><dd>{health?.dataset_manifest_present ? '文件存在' : '不可用'}</dd></div></dl></section><section className="report-section"><h2>模型资产</h2><div className="model-list">{health?.models.map(model => <div key={`${model.category}-${model.model}`}><div><strong>{model.label} / {model.category}</strong><small>PatchCore · ResNet18 · CPU</small></div><Badge value={model.available ? 'completed' : 'failed'} /><p>{model.reason}</p></div>)}</div></section><section className="report-section"><h2>常见问题</h2><details><summary>后端连接失败或请求超时</summary><p>确认后端已启动、端口未被占用，并检查前端的后端地址配置。点击“重新检查”恢复连接，已有任务可在检测历史中找回。</p></details><details><summary>模型或证据文件缺失</summary><p>检查本地模型、校准策略和参考索引是否完整。服务不会自动下载缺失资产；恢复文件后重新提交任务。</p></details><details><summary>为什么任务被标记为待人工复核？</summary><p>图片质量不合格、分数接近阈值、工具调用失败或证据冲突时都会转入人工复核。请结合原图、热图和执行轨迹查看原因。</p></details></section></>}
        <footer className="page-footer"><span>VisionInspect-Agent</span><span>研究用途 · 证据可追溯 · 候选结果不作为产线放行</span></footer>
      </main>
    </div>
  </div>;
}
createRoot(document.getElementById('root')!).render(<App />);
