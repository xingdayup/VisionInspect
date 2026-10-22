import { useEffect, useRef, useState } from 'react';
import './batch.css';

type Item = { id: string; filename: string; status: string; state: string | null; attempt: number; review: { decision: string; note: string } | null };
type Batch = { id: string; name: string; status: string; created_at: string; total: number; attempts: number; counts: Record<string, number>; items?: Item[]; upload_errors?: { filename: string; message: string }[] };
type Props = { request: <T>(path: string, init?: RequestInit) => Promise<T>; apiBase: string; category: string; agentAvailable: boolean; refresh: number; onOpen: (id: string) => void };
const names: Record<string, string> = { draft: '上传草稿', staged: '等待启动', queued: '排队中', running: '处理中', paused: '已暂停', completed: '已完成', failed: '失败', cancelled: '已取消', normal_candidate: '正常候选', anomaly_candidate: '异常候选', manual_review: '待人工复核' };
const text = (value: string | null) => value ? names[value] ?? value : '尚无结论';

export function BatchWorkspace({ request, apiBase, category, agentAvailable, refresh, onOpen }: Props) {
  const [batches, setBatches] = useState<Batch[]>([]);
  const [selected, setSelected] = useState<Batch | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState('');
  const [mode, setMode] = useState(agentAvailable ? 'agent_review' : 'baseline');
  const [error, setError] = useState('');
  const [messages, setMessages] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [uploadIndex, setUploadIndex] = useState(0);
  const [onlyReview, setOnlyReview] = useState(false);
  const [tick, setTick] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function load(id?: string) {
    const list = await request<{ items: Batch[] }>('/api/batches');setBatches(list.items);setLoaded(true);
    const target = id ?? selected?.id ?? localStorage.getItem('visioninspect-batch');
    if (target) { const batch = await request<Batch>(`/api/batches/${target}`);setSelected(batch);localStorage.setItem('visioninspect-batch', target); }
  }
  useEffect(() => { void load().catch(reason => { setError(reason.message);setLoaded(true); }); }, [refresh, tick]);
  useEffect(() => {
    if (!selected || !['queued', 'running'].includes(selected.status)) return;
    let timer: ReturnType<typeof setTimeout>;let active = true;let failures = 0;
    const poll = async () => {
      try {
        const value = await request<Batch>(`/api/batches/${selected.id}`);if (!active) return;
        setSelected(value);failures = 0;
        if (!['queued', 'running'].includes(value.status)) { setTick(n => n + 1);return; }
      } catch (reason) { if (!active) return;failures++;if (failures >= 3) { setError((reason as Error).message);return; } }
      timer = setTimeout(() => void poll(), 1500);
    };
    timer = setTimeout(() => void poll(), 1000);
    return () => { active = false;clearTimeout(timer); };
  }, [selected?.id, selected?.status, tick]);

  async function create(event: React.FormEvent) {
    event.preventDefault();if (!files.length || busy) return;
    setBusy(true);setMessages([]);setError('');setUploadIndex(0);const failures: string[] = [];
    try {
      const batch = await request<Batch>('/api/batches', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim() || `质检批次 ${new Date().toLocaleString('zh-CN')}`, category, model: 'patchcore', mode }) });
      setSelected(batch);localStorage.setItem('visioninspect-batch', batch.id);
      for (let index = 0;index < files.length;index++) {
        const file = files[index];setUploadIndex(index + 1);
        try { const body = new FormData();body.append('file', file);await request(`/api/batches/${batch.id}/items`, { method: 'POST', body }); }
        catch (reason) { failures.push(`${file.name}：${(reason as Error).message}`);setMessages([...failures]); }
      }
      await load(batch.id);setFiles([]);if (input.current) input.current.value = '';
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  }
  async function action(value: string) {
    if (!selected || busy) return;setBusy(true);setError('');
    try { const batch = await request<Batch>(`/api/batches/${selected.id}/actions/${value}`, { method: 'POST' });setSelected(batch);setTick(n => n + 1); }
    catch (reason) { setError((reason as Error).message); } finally { setBusy(false); }
  }
  async function retry(id: string) {
    setBusy(true);setError('');
    try { await request(`/api/tasks/${id}/retry`, { method: 'POST' });await load(); }
    catch (reason) { setError((reason as Error).message); } finally { setBusy(false); }
  }
  const items = selected?.items ?? [];
  const reviewItems = items.filter(item => (item.review?.decision ?? item.state) === 'manual_review');
  const visible = onlyReview ? reviewItems : items;
  const finished = selected ? ['completed', 'failed', 'cancelled'].reduce((sum, key) => sum + (selected.counts[key] ?? 0), 0) : 0;

  return <>
    {error && <div className="alert error" role="alert"><strong>批次操作未完成</strong><span>{error}</span><button onClick={() => { setError('');setTick(n => n + 1); }}>重新读取</button></div>}
    <section className="batch-create panel"><div className="section-heading"><h2>新建批量质检</h2><span>逐张处理 · 每批最多 100 张</span></div>
      <form onSubmit={event => void create(event)}>
        <div className="form-grid"><label>批次名称<input value={name} maxLength={120} onChange={event => setName(event.target.value)} placeholder="例如：瓶口样本复核 01" disabled={busy} /></label><label>工作流<select value={mode} onChange={event => setMode(event.target.value)} disabled={busy}><option value="baseline">单模型检测</option><option value="agent_review" disabled={!agentAvailable}>Agent 多步复核{!agentAvailable ? '（GPU 服务未配置）' : ''}</option></select></label></div>
        <div className="batch-file-row"><input ref={input} type="file" multiple accept="image/png,image/jpeg,image/webp" aria-label="选择批量检测图片" disabled={busy} onChange={event => { const chosen = Array.from(event.target.files ?? []);if (chosen.length > 100) { setError('每批最多选择 100 张，请重新选择。');event.target.value = '';setFiles([]);return; }setFiles(chosen);setError(''); }} /><span>{files.length ? `已选择 ${files.length} 张 · ${(files.reduce((sum, file) => sum + file.size, 0) / 1024 ** 2).toFixed(1)} MiB` : '支持 PNG、JPEG、WebP，单张最多 10 MiB'}</span></div>
        {files.length !== new Set(files.map(file => `${file.name}:${file.size}`)).size && <p className="caption">存在同名同大小文件，仍会独立提交，请核对是否重复。</p>}
        {busy && files.length > 0 && <div aria-live="polite"><progress max={files.length} value={uploadIndex} /><p>正在逐张上传 {uploadIndex} / {files.length}</p></div>}
        <div className="submit-row"><span>类别：{category} · 上传完成后手动启动批次</span><button className="primary" disabled={busy || !files.length} type="submit">{busy ? '正在上传…' : '上传并创建草稿'}</button></div>
      </form>
      {messages.length > 0 && <details open className="upload-errors"><summary>{messages.length} 个文件上传失败，成功项已保留</summary><ul>{messages.map((message, index) => <li key={index}>{message}</li>)}</ul></details>}
    </section>
    <section className="history-section"><div className="section-heading"><h2>批次记录</h2><button className="text-button" onClick={() => setTick(n => n + 1)}>刷新批次</button></div>
      {!loaded ? <div className="skeleton" aria-label="正在读取批次" /> : batches.length ? <div className="table-scroll"><table><caption className="sr-only">批次记录</caption><thead><tr><th>批次</th><th>状态</th><th>图片数</th><th>待人工</th><th>创建时间</th><th>操作</th></tr></thead><tbody>{batches.map(batch => <tr key={batch.id} className={selected?.id === batch.id ? 'selected-row' : ''}><td><strong>{batch.name}</strong><small>{batch.id.slice(0, 12)}</small></td><td>{text(batch.status)}</td><td>{batch.total}</td><td>{batch.counts.manual_review ?? 0}</td><td>{new Date(batch.created_at).toLocaleString('zh-CN')}</td><td><button className="text-button" onClick={() => { setOnlyReview(false);void load(batch.id).catch(reason => setError(reason.message)); }}>打开批次</button></td></tr>)}</tbody></table></div> : <p className="empty-note">选择多张图片创建首个批次，上传与检测记录会保存在服务端。</p>}
    </section>
    {selected && <section className="batch-detail evidence-section" aria-live="polite"><div className="section-heading"><div><p className="eyebrow">BATCH / {selected.id.slice(0, 12)}</p><h2>{selected.name}</h2></div><span>{text(selected.status)} · 已结束 {finished} / {selected.total}</span></div>
      <progress max={Math.max(1, selected.total)} value={finished} aria-label="批次处理进度" />
      <div className="batch-actions">
        {['draft', 'paused', 'completed'].includes(selected.status) && <button className="primary" disabled={busy || !(selected.counts.staged + selected.counts.queued + selected.counts.running)} onClick={() => void action(selected.status === 'draft' ? 'start' : 'resume')}>{selected.status === 'draft' ? '开始批次' : '继续待执行项'}</button>}
        {['queued', 'running'].includes(selected.status) && <button className="secondary" disabled={busy} onClick={() => void action('pause')}>当前图片结束后暂停</button>}
        {!['completed', 'cancelled'].includes(selected.status) && <button className="secondary" disabled={busy} onClick={() => void action('cancel')}>取消未执行项</button>}
        <a className="button secondary" href={`${apiBase}/api/batches/${selected.id}/report`} download>导出批次 JSON</a><a className="button secondary" href={`${apiBase}/api/batches/${selected.id}/report?format=csv`} download>导出 CSV</a>
      </div>
      <p className="caption">正常候选 {selected.counts.normal_candidate} · 异常候选 {selected.counts.anomaly_candidate} · 待人工 {selected.counts.manual_review} · 失败 {selected.counts.failed} · 已人工记录 {selected.counts.human_reviewed}。候选数量不代表产品合格率。</p>
      {(selected.upload_errors?.length ?? 0) > 0 && <details className="upload-errors"><summary>上传失败记录（{selected.upload_errors!.length}）</summary><ul>{selected.upload_errors!.map((item, index) => <li key={index}>{item.filename}：{item.message}</li>)}</ul></details>}
      <div className="batch-review-toolbar"><label className="checkbox-label"><input type="checkbox" checked={onlyReview} onChange={event => setOnlyReview(event.target.checked)} />仅显示待人工复核（{reviewItems.length}）</label>{reviewItems.length > 0 && <button className="secondary" onClick={() => onOpen(reviewItems[0].id)}>开始逐项复核</button>}</div>
      {visible.length ? <div className="table-scroll"><table><caption className="sr-only">批次图片与复核进度</caption><thead><tr><th>图片</th><th>处理状态</th><th>当前结论</th><th>来源 / 尝试</th><th>操作</th></tr></thead><tbody>{visible.map(item => <tr key={item.id}><td><strong>{item.filename}</strong><small>{item.id.slice(0, 12)}</small></td><td>{text(item.status)}</td><td>{text(item.review?.decision ?? item.state)}</td><td>{item.review ? '人工记录' : '机器流程'} · 第 {item.attempt} 次</td><td><button className="text-button" onClick={() => onOpen(item.id)}>查看证据</button>{item.status === 'failed' && ['paused', 'completed'].includes(selected.status) && <button className="text-button" disabled={busy} onClick={() => void retry(item.id)}>创建重试</button>}</td></tr>)}</tbody></table></div> : <p className="empty-note">{onlyReview ? '本批次当前没有待人工复核项。' : '草稿尚无成功上传的图片。'}</p>}
      {selected.attempts > selected.total && <p className="caption">共 {selected.attempts} 次执行尝试，汇总按每张图片的最新尝试计算；原失败记录保留在检测历史和批次 JSON 中。</p>}
    </section>}
  </>;
}
