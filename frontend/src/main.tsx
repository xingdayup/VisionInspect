import { useEffect, useRef, useState, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import './style.css';
import { BatchWorkspace } from './BatchWorkspace';

type View = 'workbench' | 'batches' | 'evidence' | 'trace' | 'history' | 'report' | 'system';
type Model = { category: string; label: string; model: string; available: boolean; reason: string };
type Health = {
  cuda_available: boolean;
  status: string;
  device: string;
  models: Model[];
  inference_service: boolean;
  agent_review_available?: boolean;
  weights_present: boolean;
  dataset_manifest_present: boolean;
  worker_concurrency: number;
};
type Trace = {
  index: number;
  tool: string;
  reason: string;
  input: unknown;
  output?: unknown;
  status: string;
  seconds?: number;
  attempt: number;
  error?: { code: string; message: string };
};
type Detection = {
  score: number;
  threshold: number;
  model_version: string;
  elapsed_seconds: number;
  anomaly_map_path: string;
  evidence: Record<string, string>;
};
type Region = {
  id: string;
  crop_path: string;
  pixels: number;
  maximum_score: number;
  box: number[];
};
type Initialization = {
  status: string;
  seconds: number;
  limit_seconds: number;
  error?: { code: string; message: string };
};
type InspectionReport = {
  result: { state: string; reasons: string[]; evidence_paths: string[]; rule_version?: string };
  outputs: {
    quality?: { valid: boolean; issues: string[]; width: number; height: number };
    detection?: Detection;
    references?: { reference_paths: string[]; similarities: number[]; source_split: string };
    regions?: { regions: Region[]; anomaly_area_ratio: number };
  };
  trace: Trace[];
  seconds: number;
  tool_calls: number;
  vlm_calls: number;
  monetary_cost: number;
  initialization?: Initialization | null;
  stop_reason: string;
  policy_version: string;
  profile_digest: string;
  checkpoint_sha256: string;
  manifest_digest: string;
  notice: string;
  errors: string[];
};
type Candidate = { state: string; reasons: string[]; source?: string };
type Report = InspectionReport & {
  timings?: {
    queued_seconds: number;
    gpu_prepare_seconds: number;
    worker_seconds: number;
    detector_initialization_seconds: number;
    agent_seconds: number;
    total_processing_seconds: number;
  };
  initial_result?: Candidate;
  effective_result?: Candidate;
  agent_review?: {
    status: string;
    accepted?: boolean;
    calls: number;
    reasons?: string[];
    rounds: {
      round: number;
      view: unknown[];
      summary?: string;
      action?: string;
      input_images?: { id: string; path: string; sha256: string }[];
      observations?: { text: string; evidence_ids: string[] }[];
    }[];
  };
};
type Runtime = {
  pid?: number;
  idle_ttl_seconds?: number;
  gpu?: {
    enabled: boolean;
    ready: boolean;
    loading: boolean;
    pid: number | null;
    idle_ttl_seconds: number;
    failure: string | null;
    peak_gpu_memory_mib: number | null;
  };
};
type Review = { decision: string; note: string; recorded_at: string };
type Task = {
  batch_id?: string;
  id: string;
  created_at: string;
  updated_at: string;
  status: string;
  category: string;
  model: string;
  mode: string;
  filename: string;
  state: string | null;
  error: string | null;
  seconds: number | null;
  review: Review | null;
  progress?: { stage: string; trace: Trace[] };
  report?: Report;
};
type History = { items: Task[]; total: number; limit: number; offset: number };
const apiBase = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
const nav: { key: View; title: string; path: string }[] = [
  {
    key: 'workbench',
    title: '质检工作台',
    path: 'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
  },
  { key: 'batches', title: '批量质检', path: 'M3 5h7l2 2h9v14H3zM7 12h10M7 16h7' },
  { key: 'evidence', title: '视觉证据', path: 'M3 4h18v16H3zM3 15l5-5 4 4 3-3 6 6M15 8h.01' },
  {
    key: 'trace',
    title: '执行轨迹',
    path: 'M6 4v16M6 6h12M6 12h8M6 18h12M3 6h.01M3 12h.01M3 18h.01',
  },
  { key: 'history', title: '检测历史', path: 'M3 12a9 9 0 1 0 3-7M3 3v5h5M12 7v5l3 2' },
  { key: 'report', title: '质检报告', path: 'M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7' },
  { key: 'system', title: '系统状态', path: 'M3 4h18v13H3zM8 21h8M12 17v4M6 11h3l2-4 3 7 2-3h2' },
];
const labels: Record<string, string> = {
  normal_candidate: '正常候选',
  anomaly_candidate: '异常候选',
  manual_review: '待人工复核',
  queued: '排队中',
  running: '检测中',
  completed: '已完成',
  failed: '失败',
  baseline: '单模型',
  inspect_image_quality: '图像质量检查',
  detect_anomaly: '异常检测',
  retrieve_normal_reference: '正常参考检索',
  analyze_suspicious_region: '疑似区域分析',
  finalize_inspection: '规则汇总',
};
function label(value: string | null) {
  return value ? (labels[value] ?? value) : '等待结果';
}
function seconds(value?: number | null) {
  return value == null ? '—' : `${value.toFixed(2)} s`;
}
Object.assign(labels, {
  agent_review: 'Agent 多步复核',
  agent_decision: 'Agent 观察与决策',
  review_evidence: 'Agent 观察与决策',
  compare_region: '局部对照检查',
  compare_reference: '切换正常参考',
  waiting_vlm: '正在准备 GPU 模型',
  staged: '等待批次启动',
  paused: '已暂停',
  cancelled: '已取消',
  initialize_model: '正在加载本地检测模型',
  model_ready: '检测模型已就绪',
  initialization_failed: '检测模型初始化失败',
});

function evidenceIssue(task: Task, pending: boolean) {
  const errors = [task.error, ...(task.report?.errors ?? [])].filter(Boolean).join('；');
  if (pending) return { title: '证据正在生成', message: '任务仍在执行，可查看实时执行轨迹。' };
  if (/timeout|超时|超过.*时限|超过耗时/.test(errors))
    return {
      title: '检测超时',
      message: '模型初始化或工具执行超过时限，本次没有有效检测结果。请查看执行轨迹后重新提交。',
    };
  if (task.report?.initialization?.status === 'failed')
    return {
      title: '模型初始化失败',
      message: errors || '本地模型加载失败，请查看执行轨迹与任务日志。',
    };
  if (errors || task.status === 'failed')
    return { title: '检测未能完成', message: errors || '任务执行失败，请查看执行轨迹或重新提交。' };
  const issues = task.report?.outputs.quality?.issues.filter(Boolean).join('；');
  if (task.report?.outputs.quality?.valid === false)
    return {
      title: '图像质量未通过',
      message: issues || '请检查图片清晰度、曝光和分辨率后重新提交。',
    };
  return { title: '检测证据不可用', message: '没有可用的模型结果，请查看执行轨迹后重新提交。' };
}
function date(value: string) {
  return new Date(value).toLocaleString('zh-CN', { hour12: false });
}
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiBase + path, {
      ...init,
      signal: init.signal
        ? AbortSignal.any([init.signal, AbortSignal.timeout(12000)])
        : AbortSignal.timeout(12000),
    });
  } catch (error) {
    throw new Error(
      error instanceof DOMException &&
      (error.name === 'TimeoutError' || error.name === 'AbortError')
        ? '连接超时，请重试或检查后端服务。'
        : '无法连接后端，请确认本地服务已启动。',
    );
  }
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const body = await response.json();
      message = typeof body.detail === 'string' ? body.detail : message;
    } catch {
      /* Keep readable HTTP status. */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
function Icon({ path, size = 19 }: { path: string; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={path} />
    </svg>
  );
}
function Badge({ value }: { value: string | null }) {
  return (
    <span className={`badge ${value ?? ''}`}>
      <span className="status-dot" />
      {label(value)}
    </span>
  );
}
function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <Icon size={32} path="M4 4h16v16H4zM8 8h8M8 12h5M8 16h3" />
      <h2>{title}</h2>
      <p>{children}</p>
      {action}
    </div>
  );
}

function App() {
  const [view, setView] = useState<View>('workbench');
  const [health, setHealth] = useState<Health | null>(null);
  const [history, setHistory] = useState<History>({ items: [], total: 0, limit: 20, offset: 0 });
  const [task, setTask] = useState<Task | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [category, setCategory] = useState('bottle');
  const [mode, setMode] = useState('agent_review');
  const [runtime, setRuntime] = useState<Runtime | null>(null);
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
  const pending =
    task?.status === 'queued' || task?.status === 'running' || task?.status === 'staged';
  const report = task?.report;
  const detection = report?.outputs?.detection;
  const trace = report?.trace ?? task?.progress?.trace ?? [];
  const categories = health?.models.filter((model) => model.model === 'patchcore') ?? [];
  const selectedModel = categories.find((model) => model.category === category);
  const selectedTitle = nav.find((item) => item.key === view)!.title;

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
  }, [view]);
  useEffect(() => {
    setReviewNote('');
    setReviewMessage('');
    setReviewDecision(task?.review?.decision ?? 'manual_review');
  }, [task?.id]);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setLoading(true);
    Promise.all([
      request<Health>('/health', { signal: controller.signal }),
      request<History>(
        `/api/tasks?limit=20&offset=${offset}&state=${filter}&q=${encodeURIComponent(query)}`,
        { signal: controller.signal },
      ),
      request<Runtime>('/api/runtime', { signal: controller.signal }),
    ])
      .then(([state, items, activeRuntime]) => {
        setHealth(state);
        setHistory(items);
        setRuntime(activeRuntime);
        setError('');
      })
      .catch((reason) => {
        if (active) {
          setHealth(null);
          setError(String(reason.message));
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [refresh, filter, query, offset]);

  useEffect(() => {
    const stored = localStorage.getItem('visioninspect-task');
    if (stored)
      request<Task>(`/api/tasks/${stored}`)
        .then(setTask)
        .catch(() => localStorage.removeItem('visioninspect-task'));
  }, []);
  useEffect(() => {
    if (!file) {
      setPreview('');
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (!task || !pending || task.status === 'staged' || pollFailed) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let failures = 0;
    async function poll() {
      try {
        const value = await request<Task>(`/api/tasks/${task!.id}`, { signal: controller.signal });
        setTask(value);
        failures = 0;
        if (value.status === 'completed' || value.status === 'failed') {
          setRefresh((n) => n + 1);
          setView('evidence');
          return;
        }
      } catch (reason) {
        if (controller.signal.aborted) return;
        failures++;
        if (failures >= 3) {
          setPollFailed(true);
          setError(String((reason as Error).message));
          return;
        }
      }
      timer = setTimeout(() => void poll(), 1000);
    }
    timer = setTimeout(() => void poll(), 500);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [task?.id, pending, pollFailed]);

  function chooseFile(value?: File) {
    if (!value) return;
    setError('');
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(value.type)) {
      setFile(null);
      setError('请选择 PNG、JPEG 或 WebP 图像。');
      return;
    }
    if (value.size > 10 * 1024 * 1024) {
      setFile(null);
      setError('图像超过 10 MiB 上限，请选择较小的文件。');
      return;
    }
    if (value.size === 0) {
      setFile(null);
      setError('文件为空，请重新选择。');
      return;
    }
    setFile(value);
  }
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!file || submitting || pending) return;
    setSubmitting(true);
    setError('');
    setReviewMessage('');
    setPollFailed(false);
    const body = new FormData();
    body.append('file', file);
    body.append('category', category);
    body.append('model', 'patchcore');
    body.append('mode', mode);
    try {
      const value = await request<Task>('/api/tasks', { method: 'POST', body });
      setTask(value);
      localStorage.setItem('visioninspect-task', value.id);
      setRefresh((n) => n + 1);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setSubmitting(false);
    }
  }
  async function selectTask(id: string) {
    setError('');
    setReviewMessage('');
    setReviewNote('');
    setPollFailed(false);
    try {
      const value = await request<Task>(`/api/tasks/${id}`);
      setTask(value);
      localStorage.setItem('visioninspect-task', id);
      setView(value.status === 'queued' || value.status === 'running' ? 'workbench' : 'evidence');
    } catch (reason) {
      setError((reason as Error).message);
    }
  }
  async function saveReview(event: React.FormEvent) {
    event.preventDefault();
    if (!task || !reviewNote.trim()) return;
    setSavingReview(true);
    setReviewMessage('');
    setError('');
    try {
      await request(`/api/tasks/${task.id}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision: reviewDecision, note: reviewNote.trim() }),
      });
      setTask(await request<Task>(`/api/tasks/${task.id}`));
      setReviewMessage('复核记录已保存，原始模型结论保持可追溯。');
      setRefresh((n) => n + 1);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setSavingReview(false);
    }
  }
  async function nextReview() {
    if (!task?.batch_id) return;
    try {
      const batch = await request<{ items: Task[] }>(`/api/batches/${task.batch_id}`);
      const next = batch.items.find(
        (item) => item.id !== task.id && (item.review?.decision ?? item.state) === 'manual_review',
      );
      if (next) await selectTask(next.id);
      else {
        setReviewMessage('本批次没有其他待人工复核项。');
      }
    } catch (reason) {
      setError((reason as Error).message);
    }
  }
  function asset(path: string) {
    path = path.replace(
      /^vlm_input\/round_([1-3])\/image_([1-3])\.png$/,
      'vlm_round_$1_image_$2.png',
    );
    return `${apiBase}/api/tasks/${task!.id}/artifacts/${encodeURIComponent(path.split('/').pop()!)}`;
  }
  function image(path: string, alt: string, className = '') {
    return (
      <img
        className={className}
        src={asset(path)}
        alt={alt}
        loading="lazy"
        onError={(event) => {
          event.currentTarget.alt = `${alt}（证据无法读取）`;
          event.currentTarget.classList.add('image-error');
        }}
      />
    );
  }
  function taskTable(items: Task[], compact = false) {
    return (
      <div className="table-scroll">
        <table>
          <caption className="sr-only">真实检测任务记录</caption>
          <thead>
            <tr>
              <th>文件 / 任务</th>
              <th>候选状态</th>
              {!compact && (
                <>
                  <th>工作流</th>
                  <th>耗时</th>
                </>
              )}
              <th>提交时间</th>
              <th>
                <span className="sr-only">打开任务</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id}>
                <td>
                  <strong>{item.filename}</strong>
                  <small>{item.id.slice(0, 12)}</small>
                </td>
                <td>
                  <Badge value={item.status === 'completed' ? item.state : item.status} />
                  {item.review && <small>已有人工复核</small>}
                </td>
                {!compact && (
                  <>
                    <td>{label(item.mode)}</td>
                    <td>{seconds(item.seconds)}</td>
                  </>
                )}
                <td className="muted nowrap">{date(item.created_at)}</td>
                <td>
                  <button className="text-button" onClick={() => void selectTask(item.id)}>
                    查看<span aria-hidden="true"> ↗</span>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  const noTask = (
    <Empty
      title="先选择一条检测任务"
      action={<button onClick={() => setView('workbench')}>前往工作台</button>}
    >
      上传产品图像，或从检测历史中打开任务，即可查看证据、执行轨迹和报告。
    </Empty>
  );
  const issue = task ? evidenceIssue(task, Boolean(pending)) : null;

  return (
    <div className="app-shell">
      <a className="skip" href="#content">
        跳至主要内容
      </a>
      <aside className="sidebar">
        <a
          href="#"
          className="brand"
          onClick={(event) => {
            event.preventDefault();
            setView('workbench');
          }}
        >
          <span className="brand-mark">
            <Icon path="M5 3H3v5M19 3h2v5M3 16v5h5M21 16v5h-5M7 12l3 3 7-7" size={25} />
          </span>
          <span>
            VisionInspect<small>视觉质检工作台</small>
          </span>
        </a>
        <div className="nav-label">工作空间</div>
        <nav aria-label="主要导航">
          {nav.map((item) => (
            <button
              key={item.key}
              className={view === item.key ? 'nav-item selected' : 'nav-item'}
              aria-current={view === item.key ? 'page' : undefined}
              onClick={() => setView(item.key)}
            >
              <Icon path={item.path} />
              {item.title}
              {item.key === 'evidence' &&
                (task?.review?.decision ?? task?.state) === 'manual_review' && (
                  <span className="nav-count">1</span>
                )}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span className={`connection ${health ? 'online' : ''}`}>
            <span className="status-dot" />
            {loading ? '正在连接' : health ? '本地服务已连接' : '后端未连接'}
          </span>
          <small>
            {!health?.cuda_available
              ? 'CUDA 未就绪'
              : health.agent_review_available
                ? 'GPU 检测与复核'
                : 'GPU 检测 · 复核资产未配置'}
          </small>
          <p>
            研究质检
            <br />
            结论保留证据与来源
          </p>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div>
            <span>工作空间</span>
            <span className="slash">/</span>
            <strong>{selectedTitle}</strong>
          </div>
          <div className="topbar-right">
            <span className="outline-tag">
              {health?.agent_review_available ? 'CUDA / vLLM' : 'CUDA / FP32'}
            </span>
            <button
              className="icon-button"
              title="刷新连接和记录"
              aria-label="刷新连接和记录"
              onClick={() => {
                setRefresh((n) => n + 1);
                setPollFailed(false);
              }}
            >
              <Icon path="M20 8a8 8 0 1 0 0 8M20 3v5h-5" />
            </button>
          </div>
        </header>
        <main id="content" tabIndex={-1}>
          <div className="page-heading">
            <div>
              <p className="eyebrow">
                {view === 'workbench' ? 'INSPECTION' : 'VISIONINSPECT / LOCAL'}
              </p>
              <h1>{selectedTitle}</h1>
              <p>
                {view === 'workbench'
                  ? '上传产品图像，从模型判断到证据复核。'
                  : view === 'batches'
                    ? '逐张检测、按需复核，追踪每个批次的处理结果。'
                    : view === 'history'
                      ? '追溯每次检测的输入、证据与人工记录。'
                      : view === 'system'
                        ? '检查本地服务与模型资产的可用性。'
                        : task
                          ? `${task.filename} · ${task.id.slice(0, 12)}`
                          : '选择任务后查看完整检测记录。'}
              </p>
            </div>
            {task && ['evidence', 'trace', 'report'].includes(view) && (
              <button className="secondary" onClick={() => setView('history')}>
                切换任务
              </button>
            )}
          </div>
          {error && (
            <div className="alert error" role="alert">
              <strong>操作未完成</strong>
              <span>{error}</span>
              <button
                className="text-button"
                onClick={() => {
                  setError('');
                  setRefresh((n) => n + 1);
                  setPollFailed(false);
                }}
              >
                重试连接
              </button>
            </div>
          )}
          {view === 'workbench' && (
            <>
              {health && !health.cuda_available && (
                <div className="alert" role="status">
                  <strong>CUDA 尚不可用</strong>
                  <span>检测需要 NVIDIA GPU，请在服务器核对 CUDA PyTorch 和驱动环境。</span>
                  <button onClick={() => setView('system')}>查看系统状态</button>
                </div>
              )}
              <div className="workbench-grid">
                <section className="panel upload-panel" aria-labelledby="upload-heading">
                  <div className="section-heading">
                    <h2 id="upload-heading">新建检测</h2>
                    <span className="step-label">01 / 选择图像</span>
                  </div>
                  <form onSubmit={(event) => void submit(event)}>
                    <div
                      className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
                      onDragOver={(event) => {
                        event.preventDefault();
                        if (!pending) setDragging(true);
                      }}
                      onDragLeave={() => setDragging(false)}
                      onDrop={(event) => {
                        event.preventDefault();
                        setDragging(false);
                        if (!pending) chooseFile(event.dataTransfer.files[0]);
                      }}
                    >
                      {preview ? (
                        <>
                          <img src={preview} alt="待检测图像预览" className="upload-preview" />
                          <strong className="filename">{file?.name}</strong>
                          <small>{((file?.size ?? 0) / 1024).toFixed(0)} KB</small>
                          <button
                            type="button"
                            className="text-button"
                            disabled={pending || submitting}
                            onClick={() => fileInput.current?.click()}
                          >
                            更换图像
                          </button>
                        </>
                      ) : (
                        <>
                          <div className="upload-symbol">
                            <Icon path="M12 16V4M7 9l5-5 5 5M4 16v4h16v-4" size={29} />
                          </div>
                          <h3>将产品图像拖到这里</h3>
                          <p>或从本机选择一张图片</p>
                          <button
                            type="button"
                            className="secondary"
                            onClick={() => fileInput.current?.click()}
                          >
                            选择图像
                          </button>
                          <small>PNG、JPEG、WebP · 最大 10 MiB</small>
                        </>
                      )}
                      <input
                        ref={fileInput}
                        type="file"
                        accept="image/png,image/jpeg,image/webp"
                        className="sr-only"
                        aria-label="选择待检测图像"
                        disabled={pending || submitting}
                        onChange={(event) => chooseFile(event.target.files?.[0])}
                      />
                    </div>
                    <div className="form-grid">
                      <label>
                        产品类别
                        <select
                          value={category}
                          onChange={(event) => setCategory(event.target.value)}
                          disabled={pending || submitting || !categories.length}
                        >
                          {categories.length ? (
                            categories.map((model) => (
                              <option key={model.category} value={model.category}>
                                {model.label} / {model.category}
                              </option>
                            ))
                          ) : (
                            <option value="bottle">正在读取类别</option>
                          )}
                        </select>
                      </label>
                      <label>
                        检测模型
                        <select value="patchcore" disabled>
                          <option value="patchcore">PatchCore · ResNet18</option>
                        </select>
                      </label>
                    </div>
                    <label>
                      工作流
                      <select
                        value={mode}
                        onChange={(event) => setMode(event.target.value)}
                        disabled={pending || submitting}
                      >
                        <option value="agent_review" disabled={!health?.agent_review_available}>
                          Agent 多步复核
                          {!health?.agent_review_available ? '（复核资产未就绪）' : ''}
                        </option>
                        <option value="baseline">单模型 · 阈值判断</option>
                      </select>
                    </label>
                    <p className="caption">
                      {mode === 'agent_review'
                        ? '不确定样本由 Agent 查看整体、局部与正常参考，证据不足时转人工。'
                        : '使用 CUDA PatchCore 和冻结阈值生成候选结果。'}
                    </p>
                    {selectedModel && !selectedModel.available && (
                      <p className="inline-error" role="alert">
                        {selectedModel.reason}
                      </p>
                    )}
                    <div className="submit-row">
                      <span>由当前部署实例处理图像</span>
                      <button
                        className="primary"
                        type="submit"
                        disabled={
                          !file ||
                          !health?.inference_service ||
                          !selectedModel?.available ||
                          (mode === 'agent_review' && !health?.agent_review_available) ||
                          submitting ||
                          pending
                        }
                      >
                        {submitting ? '正在上传…' : pending ? '任务处理中…' : '开始检测'}
                        <Icon path="M5 12h14M14 7l5 5-5 5" size={17} />
                      </button>
                    </div>
                  </form>
                </section>
                <aside className="workflow-notes">
                  <p className="eyebrow">检查流程</p>
                  <h2>
                    结论背后，
                    <br />
                    都有可查看的证据。
                  </h2>
                  <ol>
                    <li>
                      <span>01</span>
                      <div>
                        <strong>质量与异常检测</strong>
                        <p>先检查输入质量，再计算图像分数和像素热图。</p>
                      </div>
                    </li>
                    <li>
                      <span>02</span>
                      <div>
                        <strong>Agent 选择下一步检查</strong>
                        <p>不确定样本按需查看局部与正常参考，形成结构化观察。</p>
                      </div>
                    </li>
                    <li>
                      <span>03</span>
                      <div>
                        <strong>形成候选结果</strong>
                        <p>规则汇总结果，证据不足时转入人工复核。</p>
                      </div>
                    </li>
                  </ol>
                  <div className="note">
                    <Icon path="M12 3l9 17H3zM12 9v4M12 17h.01" />
                    <p>候选状态用于研究复核，不构成工业产线的放行授权。</p>
                  </div>
                </aside>
              </div>
              {pending && task && (
                <section className="panel progress-panel" aria-live="polite">
                  <div className="section-heading">
                    <h2>
                      {label(task.status)} · {task.filename}
                    </h2>
                    <span>{task.id.slice(0, 12)}</span>
                  </div>
                  <progress
                    max={6}
                    value={Math.min(trace.filter((item) => item.status === 'completed').length, 5)}
                  />
                  <p>
                    {label(task.progress?.stage ?? task.status)}
                    {pollFailed && ' · 轮询暂停，请重试连接'}
                  </p>
                  <button className="text-button" onClick={() => setView('trace')}>
                    查看实时执行轨迹 ↗
                  </button>
                </section>
              )}
              <section className="history-section">
                <div className="section-heading">
                  <h2>最近检测</h2>
                  <button
                    className="text-button"
                    onClick={() => {
                      setFilter('');
                      setQuery('');
                      setOffset(0);
                      setView('history');
                    }}
                  >
                    全部记录 ↗
                  </button>
                </div>
                {loading ? (
                  <div className="skeleton" aria-label="正在读取记录" />
                ) : history.items.length ? (
                  taskTable(history.items.slice(0, 5), true)
                ) : (
                  <Empty title="开始第一条检测">选择图像并提交，完成后的记录会保存在这里。</Empty>
                )}
              </section>
            </>
          )}
          {view === 'batches' && (
            <BatchWorkspace
              request={request}
              apiBase={apiBase}
              category={category}
              agentAvailable={Boolean(health?.agent_review_available)}
              refresh={refresh}
              onOpen={(id) => void selectTask(id)}
            />
          )}
          {view === 'evidence' &&
            (task ? (
              <>
                <section className="result-strip">
                  <div>
                    <Badge
                      value={
                        task.status === 'completed' || task.status === 'failed'
                          ? (task.review?.decision ?? task.state)
                          : task.status
                      }
                    />
                    <p>
                      {task.review?.note ??
                        report?.result.reasons.join('；') ??
                        task.error ??
                        '任务正在运行，证据将在完成后显示。'}
                    </p>
                    {detection && (
                      <p className="caption">
                        模型 {task.model} · {detection.model_version} · 规则{' '}
                        {report?.result.rule_version ?? report?.policy_version ?? '未记录'}
                      </p>
                    )}
                  </div>
                  <dl>
                    <div>
                      <dt>异常分数</dt>
                      <dd>{detection?.score.toFixed(4) ?? '—'}</dd>
                    </div>
                    <div>
                      <dt>冻结阈值</dt>
                      <dd>{detection?.threshold.toFixed(4) ?? '—'}</dd>
                    </div>
                    <div>
                      <dt>模型推理</dt>
                      <dd>{seconds(detection?.elapsed_seconds)}</dd>
                    </div>
                  </dl>
                </section>
                {report && (
                  <section className="decision-layers" aria-label="结论来源">
                    <div>
                      <h3>原始检测</h3>
                      <Badge value={report.initial_result?.state ?? report.result.state} />
                      <p>冻结模型分数与阈值</p>
                    </div>
                    <div>
                      <h3>Agent 复核</h3>
                      {report.agent_review?.accepted ? (
                        <>
                          <strong>
                            {report.result.state === 'normal_candidate'
                              ? '复核通过（正常候选）'
                              : '异常候选'}
                          </strong>
                          <p>{report.result.reasons.join('；')}</p>
                        </>
                      ) : (
                        <>
                          <strong>
                            {report.agent_review?.status === 'skipped'
                              ? '未触发'
                              : report.agent_review
                                ? '需人工判断'
                                : task.mode === 'baseline'
                                  ? '未启用'
                                  : '尚无复核结论'}
                          </strong>
                          <p>
                            {report.agent_review?.reasons?.join('；') ?? '没有新的多步复核结论'}
                          </p>
                        </>
                      )}
                    </div>
                    <div>
                      <h3>人工记录</h3>
                      {task.review ? (
                        <>
                          <Badge value={task.review.decision} />
                          <p>{task.review.note}</p>
                        </>
                      ) : (
                        <p>尚未记录</p>
                      )}
                    </div>
                  </section>
                )}
                {report?.agent_review && report.agent_review.rounds.length > 0 && (
                  <section className="evidence-section">
                    <h2>Agent 复核依据</h2>
                    {report.agent_review.rounds.map((round) => (
                      <div className="review-round" key={round.round}>
                        <h3>
                          第 {round.round} 轮 · {label(round.action ?? 'failed')}
                        </h3>
                        <p>{round.summary}</p>
                        {round.input_images && (
                          <div className="reference-list">
                            {round.input_images.map((item) => (
                              <figure key={item.id}>
                                {image(item.path, `Agent 实际输入：${item.id}`)}
                                <figcaption>
                                  <strong>{item.id}</strong>
                                  <small>实际请求图像 · SHA {item.sha256.slice(0, 12)}</small>
                                </figcaption>
                              </figure>
                            ))}
                          </div>
                        )}
                        <ul>
                          {round.observations?.map((observation, index) => (
                            <li key={index}>
                              {observation.text}
                              <small>证据：{observation.evidence_ids.join('、')}</small>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                    <p className="caption">结构化复核是研究判断，效果尚需独立标注数据验证。</p>
                  </section>
                )}
                {detection ? (
                  <>
                    <div className="evidence-toolbar">
                      <h2>图像对照</h2>
                      <ToggleGroup.Root
                        className="segmented"
                        type="single"
                        value={imageMode}
                        onValueChange={(value) => {
                          if (value) setImageMode(value);
                        }}
                        orientation="horizontal"
                        aria-label="右侧证据类型"
                      >
                        <ToggleGroup.Item value="overlay" aria-label="热图叠加">
                          热图叠加
                        </ToggleGroup.Item>
                        <ToggleGroup.Item value="heatmap" aria-label="异常热图">
                          异常热图
                        </ToggleGroup.Item>
                      </ToggleGroup.Root>
                    </div>
                    <div className="image-comparison">
                      <figure>
                        {image(detection.evidence.original, '检测输入原图')}
                        <figcaption>
                          <strong>原图</strong>
                          <span>
                            {report?.outputs.quality?.width} × {report?.outputs.quality?.height}
                          </span>
                        </figcaption>
                      </figure>
                      <figure>
                        {image(
                          detection.evidence[imageMode],
                          imageMode === 'overlay'
                            ? '异常热图与原图叠加'
                            : '原始异常分数的可视化热图',
                        )}
                        <figcaption>
                          <strong>{imageMode === 'overlay' ? '热图叠加' : '异常热图'}</strong>
                          <span>256 × 256</span>
                        </figcaption>
                      </figure>
                    </div>
                    <p className="caption">
                      亮色表示本图内相对高分，不等于已确认缺陷。请结合分数、冻结阈值和原图复核。
                      <a href={asset(detection.anomaly_map_path)} download>
                        下载原始异常图 .npy
                      </a>
                    </p>
                    <section className="evidence-section">
                      <div className="section-heading">
                        <h2>疑似区域</h2>
                        <span>
                          {report?.outputs.regions
                            ? `高分像素占比 ${(report.outputs.regions.anomaly_area_ratio * 100).toFixed(2)}%`
                            : '本次未触发区域工具'}
                        </span>
                      </div>
                      {report?.outputs.regions?.regions.length ? (
                        <div className="region-list">
                          {report.outputs.regions.regions.map((region) => (
                            <figure key={region.id}>
                              {image(region.crop_path, `${region.id} 疑似区域裁剪`)}
                              <figcaption>
                                <strong>{region.id}</strong>
                                <span>
                                  {region.pixels} 个像素 · 最高分 {region.maximum_score.toFixed(3)}
                                </span>
                                <small>原图框 [{region.box.join(', ')}]</small>
                              </figcaption>
                            </figure>
                          ))}
                        </div>
                      ) : (
                        <p className="muted">
                          {report?.outputs.regions
                            ? '冻结像素阈值下未形成满足大小限制的连通区域。'
                            : '当前工作流根据模型分数决定是否提取区域。'}
                        </p>
                      )}
                    </section>
                    <section className="evidence-section">
                      <div className="section-heading">
                        <h2>正常参考图</h2>
                        <span>仅来自正常训练划分</span>
                      </div>
                      {report?.outputs.references ? (
                        <div className="reference-list">
                          {report.outputs.references.reference_paths.map((path, i) => (
                            <figure key={path}>
                              {image(path, `正常训练参考图 ${i + 1}`)}
                              <figcaption>
                                <strong>参考 {i + 1}</strong>
                                <span>
                                  特征相似度 {report.outputs.references!.similarities[i].toFixed(5)}
                                </span>
                              </figcaption>
                            </figure>
                          ))}
                        </div>
                      ) : (
                        <p className="muted">
                          本次未触发参考检索。动态 Agent 仅对不确定样本调用此工具。
                        </p>
                      )}
                    </section>
                  </>
                ) : (
                  <Empty
                    title={issue!.title}
                    action={
                      <button className="secondary" onClick={() => setView('trace')}>
                        查看执行轨迹
                      </button>
                    }
                  >
                    {issue!.message}
                  </Empty>
                )}
                {task.batch_id && (
                  <div className="batch-actions">
                    <button className="secondary" onClick={() => setView('batches')}>
                      返回批次
                    </button>
                    <button className="secondary" onClick={() => void nextReview()}>
                      下一待复核项
                    </button>
                  </div>
                )}
                {['completed', 'failed'].includes(task.status) && (
                  <section className="panel review-panel">
                    <div className="section-heading">
                      <h2>人工复核记录</h2>
                      <span>保留原始模型结论</span>
                    </div>
                    {task.review && (
                      <div className="saved-review">
                        <Badge value={task.review.decision} />
                        <p>{task.review.note}</p>
                        <small>{date(task.review.recorded_at)}</small>
                      </div>
                    )}
                    <form onSubmit={(event) => void saveReview(event)}>
                      <div className="form-grid review-grid">
                        <label>
                          复核候选状态
                          <select
                            value={reviewDecision}
                            onChange={(event) => setReviewDecision(event.target.value)}
                          >
                            <option value="manual_review">继续人工复核</option>
                            <option value="normal_candidate">正常候选</option>
                            <option value="anomaly_candidate">异常候选</option>
                          </select>
                        </label>
                        <label>
                          复核说明
                          <textarea
                            rows={2}
                            maxLength={2000}
                            required
                            value={reviewNote}
                            onChange={(event) => setReviewNote(event.target.value)}
                            placeholder="记录你观察到的证据或需要进一步确认的内容"
                          />
                        </label>
                      </div>
                      <div className="submit-row">
                        <p role="status">
                          {reviewMessage || '此记录用于研究复核，不代表产线放行。'}
                        </p>
                        <button
                          type="submit"
                          className="secondary"
                          disabled={savingReview || !reviewNote.trim()}
                        >
                          {savingReview ? '保存中…' : '保存复核'}
                        </button>
                      </div>
                    </form>
                  </section>
                )}
              </>
            ) : (
              noTask
            ))}
          {view === 'trace' &&
            (task ? (
              <>
                <div className="trace-summary">
                  <Badge value={task.status} />
                  <span>{trace.length} 次工具调用</span>
                  <span>端到端 {seconds(report?.seconds)}</span>
                  <span>费用 $0 · 本地执行</span>
                </div>
                {trace.length ? (
                  <ol className="trace-list">
                    {trace.map((item) => (
                      <li key={item.index}>
                        <details open={item.status === 'failed' || item.status === 'running'}>
                          <summary>
                            <span className="trace-number">
                              {String(item.index).padStart(2, '0')}
                            </span>
                            <span className="trace-title">
                              <strong>{label(item.tool)}</strong>
                              <small>{item.reason}</small>
                            </span>
                            <Badge value={item.status} />
                            <span className="trace-duration">{seconds(item.seconds)}</span>
                          </summary>
                          <div className="trace-detail">
                            <p>
                              <strong>调用原因：</strong>
                              {item.reason}
                            </p>
                            <p>
                              第 {item.attempt} 次尝试
                              {item.error && ` · ${item.error.code}: ${item.error.message}`}
                            </p>
                            <div className="trace-io">
                              <div>
                                <h3>输入</h3>
                                <pre>{JSON.stringify(item.input, null, 2)}</pre>
                              </div>
                              <div>
                                <h3>输出</h3>
                                <pre>
                                  {JSON.stringify(
                                    item.output ?? item.error ?? '工具执行中…',
                                    null,
                                    2,
                                  )}
                                </pre>
                              </div>
                            </div>
                          </div>
                        </details>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <Empty title="任务等待执行">
                    任务已进入本地队列，开始运行后会逐步显示真实调用记录。
                  </Empty>
                )}
                <p className="caption">
                  视觉工具最多 8 次，Agent 模型最多 6 次（含重试）；单图执行最多 180 秒，GPU
                  冷启动单独计时。工具失败或证据冲突将转为人工复核。
                </p>
              </>
            ) : (
              noTask
            ))}
          {view === 'trace' && report?.timings && (
            <p className="caption">
              排队等待 {seconds(report.timings.queued_seconds)} · GPU 准备{' '}
              {seconds(report.timings.gpu_prepare_seconds)} · 检测模型加载{' '}
              {seconds(report.timings.detector_initialization_seconds)} · Agent 复核{' '}
              {seconds(report.timings.agent_seconds)} · 处理总计{' '}
              {seconds(report.timings.total_processing_seconds)}
              。排队时间包含批次暂停时间，处理时间包含加载和证据生成。
            </p>
          )}
          {view === 'trace' && report?.initialization && (
            <p className="caption">
              本任务模型初始化 {seconds(report.initialization.seconds)} ·{' '}
              {label(report.initialization.status)}。初始化与下方工具执行分别计时。
            </p>
          )}
          {view === 'history' && (
            <>
              <div className="history-toolbar">
                <label className="search-label">
                  <span className="sr-only">搜索文件名或任务编号</span>
                  <Icon path="M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0" />
                  <input
                    type="search"
                    placeholder="搜索文件名或任务编号"
                    value={query}
                    onChange={(event) => {
                      setQuery(event.target.value);
                      setOffset(0);
                    }}
                  />
                </label>
                <label className="filter-label">
                  <span className="sr-only">按候选状态筛选</span>
                  <select
                    value={filter}
                    onChange={(event) => {
                      setFilter(event.target.value);
                      setOffset(0);
                    }}
                  >
                    <option value="">全部候选状态</option>
                    <option value="manual_review">待人工复核</option>
                    <option value="anomaly_candidate">异常候选</option>
                    <option value="normal_candidate">正常候选</option>
                  </select>
                </label>
                <span className="muted">共 {history.total} 条</span>
              </div>
              {loading ? (
                <div className="skeleton" />
              ) : history.items.length ? (
                taskTable(history.items)
              ) : (
                <Empty title="没有匹配的检测记录">调整筛选条件，或前往工作台上传图像。</Empty>
              )}
              <div className="pagination">
                <button
                  className="secondary"
                  disabled={!offset || loading}
                  onClick={() => setOffset((n) => Math.max(0, n - 20))}
                >
                  上一页
                </button>
                <span>第 {Math.floor(offset / 20) + 1} 页</span>
                <button
                  className="secondary"
                  disabled={offset + 20 >= history.total || loading}
                  onClick={() => setOffset((n) => n + 20)}
                >
                  下一页
                </button>
              </div>
            </>
          )}
          {view === 'report' &&
            (task ? (
              report ? (
                <>
                  <div className="report-heading">
                    <div>
                      <Badge value={report.result.state} />
                      <h2>{task.filename}</h2>
                      <p>{report.notice}</p>
                    </div>
                    <a
                      className="button primary"
                      href={`${apiBase}/api/tasks/${task.id}/report`}
                      download={`inspection-${task.id}.json`}
                    >
                      <Icon path="M12 3v12M7 10l5 5 5-5M4 17v4h16v-4" />
                      导出 JSON 报告
                    </a>
                  </div>
                  <section className="report-section">
                    <h2>判断与依据</h2>
                    <ul>
                      {report.result.reasons.map((reason, i) => (
                        <li key={i}>{reason}</li>
                      ))}
                    </ul>
                    <dl className="facts">
                      <div>
                        <dt>图像异常分数</dt>
                        <dd>{detection?.score.toFixed(6) ?? '未生成'}</dd>
                      </div>
                      <div>
                        <dt>图像阈值</dt>
                        <dd>{detection?.threshold.toFixed(6) ?? '未生成'}</dd>
                      </div>
                      <div>
                        <dt>工作流</dt>
                        <dd>{label(task.mode)}</dd>
                      </div>
                      <div>
                        <dt>工具调用 / VLM 调用</dt>
                        <dd>
                          {report.tool_calls ?? '—'} / {report.vlm_calls ?? '—'}
                        </dd>
                      </div>
                      <div>
                        <dt>任务总耗时</dt>
                        <dd>{seconds(task.seconds)}</dd>
                      </div>
                      <div>
                        <dt>提交时间</dt>
                        <dd>{date(task.created_at)}</dd>
                      </div>
                    </dl>
                  </section>
                  <section className="report-section">
                    <h2>人工复核</h2>
                    {task.review ? (
                      <>
                        <Badge value={task.review.decision} />
                        <p>{task.review.note}</p>
                        <small>{date(task.review.recorded_at)}</small>
                      </>
                    ) : (
                      <p className="muted">尚未保存人工复核记录。可在视觉证据页记录复核依据。</p>
                    )}
                  </section>
                  <details className="provenance">
                    <summary>可复现信息与完整报告</summary>
                    <dl className="facts">
                      <div>
                        <dt>模型 SHA256</dt>
                        <dd className="hash">{report.checkpoint_sha256 ?? '未加载'}</dd>
                      </div>
                      <div>
                        <dt>数据清单</dt>
                        <dd className="hash">{report.manifest_digest ?? '未加载'}</dd>
                      </div>
                      <div>
                        <dt>规则 / 策略</dt>
                        <dd>
                          {report.result.rule_version ?? report.policy_version}
                          <small className="hash">{report.profile_digest}</small>
                        </dd>
                      </div>
                    </dl>
                    <pre>{JSON.stringify(report, null, 2)}</pre>
                  </details>
                </>
              ) : (
                <Empty title={pending ? '报告尚未生成' : '报告文件不可用'}>
                  {pending
                    ? '等待当前任务完成后，可以查看并导出报告。'
                    : '任务已经结束，但报告文件缺失。请检查任务日志或重新提交图像。'}
                </Empty>
              )
            ) : (
              noTask
            ))}
          {view === 'system' && (
            <>
              <section className="system-status">
                <div className={`connection ${health ? 'online' : ''}`}>
                  <span className="status-dot" />
                  {health ? '本地服务在线' : loading ? '正在检查连接' : '后端不可用'}
                </div>
                <p>
                  PatchCore 使用 CUDA；Agent 通过本机接口调用 GPU vLLM。系统不会自动下载缺失资产。
                </p>
                <button className="secondary" onClick={() => setRefresh((n) => n + 1)}>
                  重新检查
                </button>
              </section>
              <section className="report-section">
                <h2>运行环境</h2>
                <dl className="facts">
                  <div>
                    <dt>推理设备 / 精度</dt>
                    <dd>
                      {health
                        ? health.cuda_available
                          ? 'CUDA / FP32'
                          : 'CUDA 不可用'
                        : '等待连接'}
                    </dd>
                  </div>
                  <div>
                    <dt>任务并发</dt>
                    <dd>{health?.worker_concurrency ?? '—'}</dd>
                  </div>
                  <div>
                    <dt>付费 API</dt>
                    <dd>未接入</dd>
                  </div>
                  <div>
                    <dt>GPU Agent / vLLM</dt>
                    <dd>
                      {runtime?.gpu?.loading
                        ? '正在加载'
                        : runtime?.gpu?.ready
                          ? '已就绪'
                          : health?.agent_review_available
                            ? '资产已准备，首次使用时加载'
                            : '未配置'}
                    </dd>
                  </div>
                  <div>
                    <dt>PatchCore / vLLM 空闲 TTL</dt>
                    <dd>
                      {runtime?.idle_ttl_seconds ?? '—'} / {runtime?.gpu?.idle_ttl_seconds ?? '—'}{' '}
                      秒
                    </dd>
                  </div>
                  <div>
                    <dt>GPU 服务错误</dt>
                    <dd>{runtime?.gpu?.failure ?? '无已记录错误'}</dd>
                  </div>
                  <div>
                    <dt>骨干权重</dt>
                    <dd>{health?.weights_present ? '文件存在' : '不可用'}</dd>
                  </div>
                  <div>
                    <dt>数据清单</dt>
                    <dd>{health?.dataset_manifest_present ? '文件存在' : '不可用'}</dd>
                  </div>
                </dl>
              </section>
              <section className="report-section">
                <h2>模型资产</h2>
                <div className="model-list">
                  {health?.models.map((model) => (
                    <div key={`${model.category}-${model.model}`}>
                      <div>
                        <strong>
                          {model.label} / {model.category}
                        </strong>
                        <small>PatchCore · ResNet18 · CUDA</small>
                      </div>
                      <Badge value={model.available ? 'completed' : 'failed'} />
                      <p>{model.reason}</p>
                    </div>
                  ))}
                </div>
              </section>
              <section className="report-section">
                <h2>常见问题</h2>
                <details>
                  <summary>后端连接失败或请求超时</summary>
                  <p>
                    确认后端已启动、端口未被占用，并检查前端的后端地址配置。点击“重新检查”恢复连接，已有任务可在检测历史中找回。
                  </p>
                </details>
                <details>
                  <summary>模型或证据文件缺失</summary>
                  <p>
                    检查本地模型、校准策略和参考索引是否完整。服务不会自动下载缺失资产；恢复文件后重新提交任务。
                  </p>
                </details>
                <details>
                  <summary>为什么任务被标记为待人工复核？</summary>
                  <p>
                    图片质量不合格、分数接近阈值、工具调用失败或证据冲突时都会转入人工复核。请结合原图、热图和执行轨迹查看原因。
                  </p>
                </details>
              </section>
            </>
          )}
          <footer className="page-footer">
            <span>VisionInspect-Agent</span>
            <span>研究用途 · 证据可追溯 · 候选结果不作为产线放行</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
createRoot(document.getElementById('root')!).render(<App />);
