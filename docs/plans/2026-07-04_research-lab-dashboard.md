# SAGA Research Lab Dashboard

**Created:** 2026-07-04
**Status:** Planning
**Scope:** Phase 2 UI — Interactive "Living Research Lab" Dashboard

---

## 1. Vision

A React-like interactive dashboard that feels like a **high-tech research laboratory** — not a passive monitoring tool, but a living, breathing interface where researchers can watch their AI ensemble learn, detect anomalies, and make decisions in real-time.

**Core Metaphor:** The UI is a microscope into the mind of the AI system. Every prompt is a specimen, every training step is an evolution, every anomaly is a discovery.

### Design Principles

1. **Immediate Feedback** — Every action produces visible results within milliseconds
2. **Progressive Disclosure** — Overview first, details on demand
3. **Game-Like Engagement** — Progress bars, achievements, visual rewards
4. **Research-Grade Persistence** — Every metric, every decision, every prompt is stored for later analysis
5. **Accessibility First** — Keyboard navigation, screen reader support, high contrast modes
6. **Internationalization Ready** — English as SSOT, translation infrastructure from day one

---

## 2. Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Svelte 5)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Pipeline │ │  Live    │ │ Anomaly  │ │ Script   │  │
│  │ Overview │ │  Feed    │ │ Monitor  │ │ Control  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ SSE + REST
┌────────────────────────┴────────────────────────────────┐
│              Python Backend (FastAPI + SQLite)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Process  │ │  Event   │ │ Metric   │ │ Storage  │  │
│  │ Manager  │ │  Stream  │ │ Collector│ │ (SQLite) │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ subprocess + file I/O
┌────────────────────────┴────────────────────────────────┐
│                   SAGA Pipeline Scripts                  │
│  00_smoke_test │ 02_train_alignment │ 03_train_router   │
│  04_train_ae   │ 06_train_poisoned  │ 07_meta_model     │
│  08_eval       │ 08b_answer_level   │ 10_full_eval      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Frontend** | Svelte 5 + Tailwind CSS 4 | Reactive, minimal bundle, excellent for dashboards |
| **Charts** | Chart.js + svelte-chartjs | Simple, live-update capable, accessible |
| **Toasts** | svelte-french-toast | Non-intrusive notifications for anomalies |
| **Backend** | Python FastAPI | Async, native Python integration, excellent WebSocket/SSE support |
| **Database** | SQLite + SQLAlchemy | Persistent storage for research history, zero-config deployment |
| **Event Stream** | Server-Sent Events (SSE) | Simpler than WebSocket for unidirectional data flow |
| **Process Management** | Python subprocess + asyncio | Start/Pause/Stop all scripts |
| **File Watching** | watchdog (Python) | Real-time log file monitoring |
| **Testing (Backend)** | pytest + httpx | Async test client for FastAPI |
| **Testing (Frontend)** | Vitest + Testing Library | Svelte component testing |

### 2.3 Directory Structure

```
ui/
├── package.json
├── svelte.config.js
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
├── src/
│   ├── app.html
│   ├── app.css                        # Tailwind + Custom CSS (Glow, Pulse)
│   ├── lib/
│   │   ├── components/
│   │   │   ├── pipeline/
│   │   │   │   ├── PipelineView.svelte      # Full pipeline overview
│   │   │   │   ├── ScriptCard.svelte         # Individual script card
│   │   │   │   └── DependencyGraph.svelte    # Script dependencies
│   │   │   ├── live/
│   │   │   │   ├── PromptFeed.svelte         # Live prompt stream
│   │   │   │   ├── MetricGauge.svelte        # Analog gauge component
│   │   │   │   ├── LossChart.svelte          # Live loss chart
│   │   │   │   └── AccuracyBar.svelte        # Progress bar
│   │   │   ├── anomaly/
│   │   │   │   ├── AnomalyPanel.svelte       # Red warning + details
│   │   │   │   ├── ScoreHeatmap.svelte       # Anomaly scores heatmap
│   │   │   │   └── DetectionTimeline.svelte  # Detection timeline
│   │   │   ├── control/
│   │   │   │   ├── ScriptControls.svelte     # Start/Pause/Stop
│   │   │   │   ├── ParameterPanel.svelte     # CLI arg editor
│   │   │   │   └── LogViewer.svelte          # Scrollable log overlay
│   │   │   └── shared/
│   │   │       ├── StatusBadge.svelte        # Running/Success/Failed
│   │   │       ├── GlowCard.svelte           # Neon glow card frame
│   │   │       ├── PulseRing.svelte          # Pulsing ring for activity
│   │   │       ├── Toast.svelte              # Toast notification wrapper
│   │   │       └── SkipLink.svelte           # Accessibility skip link
│   │   ├── stores/
│   │   │   ├── pipeline.ts                   # Pipeline state
│   │   │   ├── metrics.ts                    # Live metrics
│   │   │   ├── events.ts                     # Event stream store
│   │   │   └── i18n.ts                       # Internationalization
│   │   ├── api/
│   │   │   ├── client.ts                     # API client (fetch + SSE)
│   │   │   └── types.ts                      # TypeScript interfaces
│   │   └── i18n/
│   │       ├── en.json                       # English (SSOT)
│   │       └── index.ts                      # i18n loader
│   └── routes/
│       ├── +page.svelte                      # Dashboard (main)
│       ├── +layout.svelte                    # Layout with navigation
│       ├── pipeline/
│       │   └── +page.svelte                  # Pipeline viewer
│       ├── live/
│       │   └── +page.svelte                  # Live feed
│       ├── anomaly/
│       │   └── +page.svelte                  # Anomaly monitor
│       └── logs/
│           └── +page.svelte                  # Log viewer
├── server/
│   ├── __init__.py
│   ├── main.py                               # FastAPI entry point
│   ├── config.py                             # Server configuration
│   ├── process_manager.py                    # Subprocess management
│   ├── event_stream.py                       # SSE event producer
│   ├── file_watcher.py                       # Log file monitoring
│   ├── metric_collector.py                   # Metric collection
│   ├── storage.py                            # SQLite storage layer
│   ├── models.py                             # Pydantic models
│   └── routes/
│       ├── __init__.py
│       ├── pipeline.py                       # Pipeline control endpoints
│       ├── metrics.py                        # Metrics endpoints
│       ├── logs.py                           # Log streaming endpoints
│       └── artifacts.py                      # Checkpoint/report endpoints
├── tests/
│   ├── conftest.py                           # Shared fixtures
│   ├── test_process_manager.py
│   ├── test_event_stream.py
│   ├── test_metric_collector.py
│   ├── test_storage.py
│   └── test_routes/
│       ├── test_pipeline.py
│       ├── test_metrics.py
│       └── test_logs.py
├── migrations/
│   └── 001_initial_schema.sql               # SQLite schema
└── static/
    └── favicon.svg                           # SAGA logo
```

---

## 3. Database Schema

### 3.1 Research History (SQLite)

```sql
-- Script execution history
CREATE TABLE script_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    script_name     TEXT NOT NULL,               -- e.g., "02_train_alignment"
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending/running/paused/completed/failed
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    parameters      JSON,                        -- CLI args used
    exit_code       INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training metrics (time-series)
CREATE TABLE training_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    step            INTEGER NOT NULL,
    epoch           INTEGER,
    metric_name     TEXT NOT NULL,               -- e.g., "train/loss"
    metric_value    REAL NOT NULL,
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation results
CREATE TABLE eval_results (
    id              PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    benchmark       TEXT NOT NULL,               -- e.g., "mmlu", "gsm8k"
    model_id        TEXT,                        -- null for ensemble
    metric_name     TEXT NOT NULL,               -- e.g., "accuracy"
    metric_value    REAL NOT NULL,
    sample_count    INTEGER,
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prompt analysis history
CREATE TABLE prompt_analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    prompt_text     TEXT NOT NULL,
    domain          TEXT,                        -- "nl" or "code"
    domain_confidence REAL,
    routing_weights JSON,                        -- {"falcon": 0.45, "qwen": 0.35, ...}
    anomaly_scores  JSON,                        -- {"mse": 0.0001, "mahalanobis": 0.23, ...}
    anomaly_detected BOOLEAN DEFAULT FALSE,
    final_answer    TEXT,
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Anomaly alerts
CREATE TABLE anomaly_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    prompt_id       INTEGER REFERENCES prompt_analyses(id),
    alert_type      TEXT NOT NULL,               -- "poisoning_detected", "threshold_exceeded"
    severity        TEXT NOT NULL DEFAULT 'warning',  -- info/warning/critical
    details         JSON,
    acknowledged    BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Checkpoint metadata
CREATE TABLE checkpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES script_runs(id),
    checkpoint_type TEXT NOT NULL,               -- "alignment", "router", "autoencoder", etc.
    file_path       TEXT NOT NULL,
    file_size       INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. Backend Implementation

### 4.1 FastAPI Server (`server/main.py`)

```python
# Core endpoints
GET  /api/health                          # Server health check
GET  /api/pipeline/status                 # Status of all scripts
GET  /api/pipeline/{run_id}               # Specific run details
POST /api/pipeline/{script_name}/start    # Start a script
POST /api/pipeline/{run_id}/pause         # Pause a script
POST /api/pipeline/{run_id}/resume        # Resume a script
POST /api/pipeline/{run_id}/stop          # Stop a script
GET  /api/metrics/stream                  # SSE: live metrics
GET  /api/metrics/history                 # Historical metrics (paginated)
GET  /api/logs/{run_id}/stream            # SSE: live log stream
GET  /api/logs/{run_id}                   # Full log text
GET  /api/anomaly/alerts                  # Recent anomaly alerts
POST /api/anomaly/alerts/{id}/ack        # Acknowledge alert
GET  /api/prompts/recent                  # Recent prompt analyses
GET  /api/artifacts/{type}                # List checkpoints/reports
GET  /api/artifacts/{type}/{name}         # Download artifact
```

### 4.2 Process Manager (`server/process_manager.py`)

```python
class ProcessManager:
    """Manages lifecycle of SAGA pipeline scripts."""

    async def start(
        self,
        script_name: str,
        params: dict[str, Any],
    ) -> ScriptRun:
        """Start a script as a subprocess."""

    async def pause(self, run_id: int) -> None:
        """Send SIGSTOP to pause a running script."""

    async def resume(self, run_id: int) -> None:
        """Send SIGCONT to resume a paused script."""

    async def stop(self, run_id: int) -> None:
        """Send SIGTERM, then SIGKILL if needed."""

    async def get_status(self, run_id: int) -> ScriptStatus:
        """Get current status of a script."""

    async def list_runs(
        self,
        status: str | None = None,
        script_name: str | None = None,
    ) -> list[ScriptRun]:
        """List all runs with optional filters."""
```

**Key Design Decisions:**
- Scripts run as independent subprocesses (not threads) for isolation
- PID tracking for process management
- stdout/stderr captured via asyncio pipes
- Graceful shutdown with SIGTERM → 5s timeout → SIGKILL

### 4.3 Event Stream (`server/event_stream.py`)

```python
class EventStream:
    """SSE event producer for real-time updates."""

    def __init__(self):
        self._clients: dict[str, set[asyncio.Queue]] = defaultdict(set)

    async def subscribe(self, channel: str) -> AsyncIterator[Event]:
        """Subscribe to a channel (e.g., 'metrics', 'logs:42', 'anomaly')."""

    async def publish(self, channel: str, event: Event) -> None:
        """Publish an event to all subscribers of a channel."""

    async def metrics_stream(self) -> AsyncIterator[dict]:
        """Stream training metrics as they arrive."""

    async def log_stream(self, run_id: int) -> AsyncIterator[str]:
        """Stream log lines for a specific run."""
```

**Channels:**
- `pipeline` — Script status changes
- `metrics:{run_id}` — Training metrics for a run
- `logs:{run_id}` — Log lines for a run
- `anomaly` — Anomaly detection events
- `prompts` — Live prompt analysis feed

### 4.4 Metric Collector (`server/metric_collector.py`)

```python
class MetricCollector:
    """Collects metrics from TensorBoard event files and JSON reports."""

    async def watch_tensorboard(self, run_dir: str) -> None:
        """Watch TensorBoard log directory for new events."""

    async def parse_json_report(self, report_path: str) -> dict:
        """Parse evaluation report JSON files."""

    async def store_metric(
        self,
        run_id: int,
        name: str,
        value: float,
        step: int,
        epoch: int | None = None,
    ) -> None:
        """Store a metric in SQLite and broadcast to SSE."""
```

### 4.5 Storage Layer (`server/storage.py`)

```python
class Storage:
    """SQLite storage for research history."""

    async def init_db(self) -> None:
        """Initialize database schema."""

    async def create_run(
        self,
        script_name: str,
        parameters: dict,
    ) -> ScriptRun:
        """Create a new script run record."""

    async def update_run_status(
        self,
        run_id: int,
        status: str,
        exit_code: int | None = None,
    ) -> None:
        """Update run status."""

    async def store_metric(
        self,
        run_id: int,
        name: str,
        value: float,
        step: int,
    ) -> None:
        """Store a training metric."""

    async def store_prompt_analysis(
        self,
        run_id: int,
        analysis: PromptAnalysis,
    ) -> None:
        """Store a prompt analysis result."""

    async def store_anomaly_alert(
        self,
        run_id: int,
        alert: AnomalyAlert,
    ) -> None:
        """Store an anomaly alert."""

    async def get_metrics_history(
        self,
        run_id: int,
        metric_name: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Get historical metrics for a run."""

    async def get_recent_prompts(
        self,
        limit: int = 100,
    ) -> list[dict]:
        """Get recent prompt analyses."""

    async def search(
        self,
        query: str,
        table: str | None = None,
    ) -> list[dict]:
        """Full-text search across all tables."""
```

---

## 5. Frontend Implementation

### 5.1 Dashboard Layout (`src/routes/+layout.svelte`)

```
┌─────────────────────────────────────────────────────────┐
│  🧬 SAGA Research Lab          [EN ▾]  [⚙️]  [❓]      │
├───────┬─────────────────────────────────────────────────┤
│       │                                                 │
│  📊   │              MAIN CONTENT AREA                  │
│  📡   │                                                 │
│  🔬   │  (Reacts to current route)                      │
│  🔴   │                                                 │
│  📋   │                                                 │
│       │                                                 │
├───────┴─────────────────────────────────────────────────┤
│  Status Bar: [🟢 System] [CPU 23%] [RAM 4.2GB] [GPU]   │
└─────────────────────────────────────────────────────────┘
```

**Navigation Items:**
- 📊 Dashboard (`/`) — Main overview
- 📡 Pipeline (`/pipeline`) — Script management
- 🔬 Live Feed (`/live`) — Real-time prompt analysis
- 🔴 Anomaly Monitor (`/anomaly`) — Security dashboard
- 📋 Logs (`/logs`) — Log viewer

### 5.2 Internationalization

**English as SSOT** — All UI text defined in `en.json` first:

```json
{
  "dashboard": {
    "title": "SAGA Research Lab",
    "status": {
      "running": "Running",
      "completed": "Completed",
      "failed": "Failed",
      "paused": "Paused"
    }
  },
  "pipeline": {
    "start": "Start",
    "pause": "Pause",
    "stop": "Stop",
    "dependencies": "Dependencies"
  },
  "anomaly": {
    "detected": "Anomaly Detected",
    "clean": "Clean",
    "triggered": "Triggered",
    "acknowledge": "Acknowledge"
  },
  "metrics": {
    "loss": "Loss",
    "accuracy": "Accuracy",
    "learningRate": "Learning Rate",
    "epoch": "Epoch",
    "step": "Step"
  }
}
```

**Translation Infrastructure:**
- i18n library loaded at app initialization
- All components use `$t('key')` for text
- Language switcher in header (persists to localStorage)
- Date/number formatting via `Intl` API

### 5.3 Accessibility (a11y)

**Requirements:**
- WCAG 2.1 AA compliance minimum
- Keyboard navigation for all interactive elements
- ARIA labels for all icons and visual indicators
- Focus management for modals and overlays
- High contrast mode support
- Reduced motion mode (respects `prefers-reduced-motion`)
- Screen reader announcements for live updates (via `aria-live`)

**Implementation:**
- Skip links for main content
- Semantic HTML throughout
- Focus rings visible on all interactive elements
- Color never the only indicator (always paired with icon/text)
- Chart components include accessible data tables

### 5.4 Key Components

#### MetricGauge.svelte — Analog Progress Indicator

```
Properties:
  value: number       (0-100)
  label: string       ("Loss", "Accuracy")
  min: number
  max: number
  color: string       (tailwind color class)
  animated: boolean

Features:
  - Smooth animation on value change
  - Color transitions (green → yellow → red)
  - Accessible: aria-valuenow, aria-valuemin, aria-valuemax
```

#### AnomalyPanel.svelte — Warning System

```
Properties:
  alerts: AnomalyAlert[]
  onAcknowledge: (id: number) => void

Features:
  - Pulsing red glow when unacknowledged alerts exist
  - Toast notification on new anomaly
  - Click to expand details
  - Keyboard accessible (Enter to acknowledge)
```

#### PromptFeed.svelte — Live Analysis Stream

```
Properties:
  maxItems: number    (default: 50)
  filter: string      ("all", "nl", "code", "anomaly")

Features:
  - Auto-scroll to bottom (with pause on hover)
  - Each prompt shows: text, domain, routing, anomaly, answer
  - Color-coded by domain (blue=NL, green=code, red=anomaly)
  - Virtual scrolling for performance
```

---

## 6. Implementation Phases

### Phase 1: Backend Core (Days 1-4)

**Goal:** Functional backend with process management and event streaming.

| Day | Task | Tests |
|-----|------|-------|
| 1 | Project setup, FastAPI skeleton, SQLite schema | `test_health_endpoint` |
| 2 | ProcessManager: start/stop/pause scripts | `test_start_script`, `test_stop_script`, `test_pause_resume` |
| 3 | EventStream: SSE subscription/publishing | `test_subscribe_publish`, `test_metrics_stream` |
| 4 | MetricCollector: TensorBoard parsing, storage | `test_parse_tensorboard`, `test_store_metric` |

**Deliverable:** Running backend with all endpoints functional.

### Phase 2: Dashboard Layout (Days 5-7)

**Goal:** Svelte app shell with navigation and responsive layout.

| Day | Task | Tests |
|-----|------|-------|
| 5 | Svelte project setup, Tailwind config, routing | Component renders |
| 6 | Layout with sidebar, header, content area | Navigation works |
| 7 | Status bar, responsive design, dark theme | Mobile layout |

**Deliverable:** Working app shell with placeholder pages.

### Phase 3: Pipeline Cards (Days 8-9)

**Goal:** Visual representation of all scripts with status control.

| Day | Task | Tests |
|-----|------|-------|
| 8 | ScriptCard component, PipelineView page | Card renders correct status |
| 9 | ScriptControls, ParameterPanel, start/stop integration | Controls trigger API calls |

**Deliverable:** Can start/stop scripts from UI.

### Phase 4: Live Metrics (Days 10-12)

**Goal:** Real-time metric visualization with charts and gauges.

| Day | Task | Tests |
|-----|------|-------|
| 10 | MetricGauge, AccuracyBar components | Gauge renders, animates |
| 11 | LossChart with Chart.js, live updates | Chart updates on new data |
| 12 | SSE integration, store updates | Data flows from backend to UI |

**Deliverable:** Live training metrics visible on dashboard.

### Phase 5: Prompt Feed (Days 13-14)

**Goal:** Real-time prompt analysis visualization.

| Day | Task | Tests |
|-----|------|-------|
| 13 | PromptFeed component, virtual scrolling | Renders list correctly |
| 14 | Live prompt stream, filtering, detail expansion | Filter works, details expand |

**Deliverable:** Live prompt feed with classification.

### Phase 6: Anomaly Panel (Days 15-17)

**Goal:** Visual anomaly detection dashboard with alerts.

| Day | Task | Tests |
|-----|------|-------|
| 15 | AnomalyPanel, StatusBadge components | Panel shows alerts |
| 16 | Toast integration, pulse animation | Toast appears on anomaly |
| 17 | ScoreHeatmap, DetectionTimeline | Heatmap renders correctly |

**Deliverable:** Full anomaly monitoring with toast notifications.

### Phase 7: Log Viewer (Days 18-19)

**Goal:** Live log streaming with search and filter.

| Day | Task | Tests |
|-----|------|-------|
| 18 | LogViewer component, SSE log stream | Logs stream correctly |
| 19 | Search, filter by level, download | Search works, download works |

**Deliverable:** Functional log viewer.

### Phase 8: Polish (Days 20-22)

**Goal:** Animations, accessibility audit, i18n finalization.

| Day | Task | Tests |
|-----|------|-------|
| 20 | Glow effects, pulse animations, transitions | Animations smooth |
| 21 | Accessibility audit, keyboard testing, screen reader | WCAG 2.1 AA pass |
| 22 | i18n finalization, translation prep, documentation | All text in en.json |

**Deliverable:** Production-ready dashboard.

---

## 7. Testing Strategy

### 7.1 TDD Approach

**Red-Green-Refactor Cycle:**
1. Write failing test
2. Implement minimum code to pass
3. Refactor for clarity/performance
4. Repeat

### 7.2 Backend Tests

```python
# Example: ProcessManager tests
@pytest.mark.asyncio
async def test_start_script(process_manager):
    """Starting a script creates a run record."""
    run = await process_manager.start("02_train_alignment", {"epochs": 3})
    assert run.status == "running"
    assert run.script_name == "02_train_alignment"
    assert run.parameters["epochs"] == 3

@pytest.mark.asyncio
async def test_stop_script(process_manager):
    """Stopping a script updates status to completed."""
    run = await process_manager.start("02_train_alignment", {})
    await process_manager.stop(run.id)
    await asyncio.sleep(1)  # Wait for process to terminate
    status = await process_manager.get_status(run.id)
    assert status == "completed"

@pytest.mark.asyncio
async def test_metric_storage(storage):
    """Metrics are stored in SQLite."""
    run = await storage.create_run("02_train_alignment", {})
    await storage.store_metric(run.id, "train/loss", 0.042, step=100)
    metrics = await storage.get_metrics_history(run.id)
    assert len(metrics) == 1
    assert metrics[0]["metric_name"] == "train/loss"
    assert metrics[0]["metric_value"] == 0.042
```

### 7.3 Frontend Tests

```typescript
// Example: MetricGauge tests
test('renders gauge with correct value', () => {
    render(MetricGauge, { value: 75, label: 'Accuracy' });
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '75');
});

test('animates value change', async () => {
    const { component } = render(MetricGauge, { value: 50 });
    component.$set({ value: 75 });
    await tick();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '75');
});
```

### 7.4 Integration Tests

```python
@pytest.mark.asyncio
async def test_end_to_end_metric_flow(client, process_manager, storage):
    """Metric flows from script to database to SSE."""
    # Start a script
    run = await process_manager.start("02_train_alignment", {})

    # Simulate metric emission
    await process_manager.emit_metric(run.id, "train/loss", 0.042, 100)

    # Verify stored in database
    metrics = await storage.get_metrics_history(run.id)
    assert any(m["metric_name"] == "train/loss" for m in metrics)

    # Verify available via API
    response = await client.get(f"/api/metrics/history?run_id={run.id}")
    assert response.status_code == 200
```

---

## 8. Accessibility Checklist

- [ ] Skip links for main content
- [ ] Semantic HTML throughout (nav, main, aside, section)
- [ ] ARIA labels for all icons and visual indicators
- [ ] Focus management for modals and overlays
- [ ] Focus rings visible on all interactive elements
- [ ] Color never the only indicator (always paired with icon/text)
- [ ] Keyboard navigation for all interactive elements
- [ ] Screen reader announcements for live updates (`aria-live="polite"`)
- [ ] High contrast mode support
- [ ] Reduced motion mode (`prefers-reduced-motion`)
- [ ] Chart components include accessible data tables
- [ ] Form inputs have associated labels
- [ ] Error messages linked to inputs via `aria-describedby`
- [ ] Modal focus trapped when open
- [ ] Escape key closes modals/overlays

---

## 9. Internationalization

### 9.1 English as SSOT

All user-facing text defined in `src/lib/i18n/en.json`:

```json
{
  "$schema": "i18n-schema",
  "meta": {
    "locale": "en",
    "version": "1.0.0"
  },
  "dashboard": {
    "title": "SAGA Research Lab",
    "subtitle": "Selective AI Generation Architecture",
    "status": {
      "running": "Running",
      "completed": "Completed",
      "failed": "Failed",
      "paused": "Paused",
      "pending": "Pending"
    }
  },
  "pipeline": {
    "title": "Pipeline Control",
    "scripts": {
      "smokeTest": "Smoke Test",
      "alignment": "Alignment Training",
      "router": "Router Training",
      "autoencoder": "Autoencoder Training",
      "calibrate": "Threshold Calibration",
      "poisoned": "Poisoned Model Training",
      "metaModel": "Meta Model Fine-tuning",
      "rewardModel": "Reward Model Training",
      "eval": "Poisoning Evaluation",
      "fullEval": "Full Evaluation"
    }
  }
}
```

### 9.2 Translation Workflow

1. All text in `en.json` (English SSOT)
2. Translations in `{locale}.json` (e.g., `de.json`, `ja.json`)
3. Missing keys fall back to English
4. Date/number formatting via `Intl.DateTimeFormat` / `Intl.NumberFormat`
5. Translator guidelines in `docs/TRANSLATION_GUIDE.md`

---

## 10. Future Extensions (Not Now)

### 10.1 Distributed Computing

- WebSocket connections to remote SAGA nodes
- Node health monitoring
- Task distribution across nodes
- Aggregate metrics from multiple machines

### 10.2 Mobile Support

- Responsive design from day one (CSS Grid/Flexbox)
- Touch-friendly controls
- Push notifications for anomalies
- Simplified mobile layout

### 10.3 Advanced Analytics

- Compare runs side-by-side
- Parameter sweep visualization
- A/B testing dashboard
- Export to Jupyter notebook

---

## 11. Success Criteria

| Criterion | Metric |
|-----------|--------|
| **Backend Response Time** | < 100ms for all endpoints |
| **SSE Latency** | < 500ms from event to UI |
| **UI Load Time** | < 2 seconds initial load |
| **Accessibility** | WCAG 2.1 AA compliant |
| **Test Coverage** | > 80% backend, > 70% frontend |
| **i18n Coverage** | 100% text in en.json |
| **Browser Support** | Chrome, Firefox, Safari (latest 2 versions) |

---

## Appendix A: Color Palette

| Name | Hex | Usage |
|------|-----|-------|
| Background | `#0a0a0f` | Main background |
| Surface | `#1a1a2e` | Card backgrounds |
| Primary | `#00d4ff` | Active elements, links |
| Success | `#00ff88` | Completed, clean |
| Warning | `#ffaa00` | Paused, attention |
| Error | `#ff0040` | Failed, anomaly |
| Purple | `#a855f7` | Metrics, charts |
| Text Primary | `#e0e0e0` | Main text |
| Text Secondary | `#808080` | Labels, captions |

## Appendix B: Animation Specifications

| Animation | Duration | Easing | Usage |
|-----------|----------|--------|-------|
| Pulse | 2s infinite | ease-in-out | Active scripts |
| Glow | 1.5s infinite | ease-in-out | Anomaly alerts |
| Fade In | 200ms | ease-out | New elements |
| Slide Up | 300ms | ease-out | Prompt feed items |
| Scale | 200ms | ease-out | Button hover |
