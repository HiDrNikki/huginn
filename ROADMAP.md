# Huginn Roadmap

This roadmap outlines the planned milestones, features, and priorities for the **Huginn** module.  
It is a contributor-facing supplement to `AGENTS.md` and is intended for both human and AI developers.

## Table of Contents

1. [Vision](#0-vision)  
   1.1 [What Huginn is](#what-huginn-is)  
   1.2 [What Huginn isn’t](#what-huginn-isnt)  
   1.3 [Why it exists (problem statement)](#why-it-exists-problem-statement)  
   1.4 [Positioning vs. Grendel & Muninn](#positioning-vs-grendel--muninn)  
   1.5 [Primary outcomes](#primary-outcomes)  
   1.6 [Non-goals (v1)](#non-goals-v1)  
   1.7 [Primary user personas](#primary-user-personas)  
   1.8 [Success criteria / KPIs](#success-criteria--kpis)  
   1.9 [Core constraints & assumptions](#core-constraints--assumptions)  
   1.10 [Interfaces (at a glance)](#interfaces-at-a-glance)  
   1.11 [Main risks & mitigation](#main-risks--mitigation)  
   1.12 [Acceptance signals](#acceptance-signals)

2. [Guiding Principles](#1-guiding-principles)  
   2.1 [Planner-Centric Design](#1-planner-centric-design)  
   2.2 [Separation of Concerns](#2-separation-of-concerns)  
   2.3 [Quantisation-First Mindset](#3-quantisation-first-mindset)  
   2.4 [Observability and Debuggability](#4-observability-and-debuggability)  
   2.5 [Security and Multi-Tenancy Awareness](#5-security-and-multi-tenancy-awareness)  
   2.6 [Topology-Aware Scheduling](#6-topology-aware-scheduling)  
   2.7 [Minimal Hot Path Logic](#7-minimal-hot-path-logic)  
   2.8 [Compatibility Without Lock-In](#8-compatibility-without-lock-in)  
   2.9 [Determinism Where It Matters](#9-determinism-where-it-matters)

3. [Milestones](#2-milestones)  
   3.1 [M0 — Core Package & CLI Skeleton](#m0--core-package--cli-skeleton)  
   3.2 [M1 — Model Registry & System Diagnostics](#m1--model-registry--system-diagnostics)  
   3.3 [M2 — Planner Engine v1 (Graph & Cost)](#m2--planner-engine-v1-graph--cost)  
   3.4 [M3 — Orchestrator & Job Manager](#m3--orchestrator--job-manager)  
   3.5 [M4 — Quantisation Integrations](#m4--quantisation-integrations)  
   3.6 [M5 — Topology-Aware Placement](#m5--topology-aware-placement)  
   3.7 [M6 — Reliability, Recovery & Integration Hardening](#m6--reliability-recovery--integration-hardening)

4. [Out of Scope (v1)](#3-out-of-scope-v1)  
   4.1 [Direct Model Training](#1-direct-model-training)  
   4.2 [Framework-Specific Optimisations in Core](#2-framework-specific-optimisations-in-core)  
   4.3 [Tight VRAM Pooling](#3-tight-vram-pooling)  
   4.4 [Cross-Node Scheduling](#4-cross-node-scheduling)

5. [Contributor Guidelines](#4-contributor-guidelines)  
   5.1 [Code Style & Structure](#1-code-style--structure)  
   5.2 [Commit & PR Practices](#2-commit--pr-practices)  
   5.3 [Testing](#3-testing)  
   5.4 [Documentation](#4-documentation)  
   5.5 [Performance Considerations](#5-performance-considerations)  
   5.6 [Observability & Logging](#6-observability--logging)  
   5.7 [Security & Safety](#7-security--safety)  
   5.8 [Communication & Coordination](#8-communication--coordination)  
   5.9 [AI Contributor Guidance](#9-ai-contributor-guidance)  
   5.10 [Merging Rules](#10-merging-rules)

6. [Reference Documents](#5-reference-documents)  
   6.1 [`AGENTS.md`](#1-agentsmd)  
   6.2 [`ARCHITECTURE.md`](#2-architecturemd)  
   6.3 [`API.md`](#3-apimd)  
   6.4 [`GRENDEL_API.md`](#4-grendel_apimd)  
   6.5 [`ROADMAP.md`](#5-roadmapmd-this-document)  
   6.6 [External References](#6-external-references)  
   6.7 [Maintenance Responsibility](#maintenance-responsibility)

---

## 0. Vision

Huginn is the **planner, orchestrator, and quantisation manager** for executing large-scale AI models across heterogeneous GPUs via **Grendel’s single logical device**.  
Where Grendel provides the illusion and mechanics of “one big GPU,” Huginn decides **what** to run, **where** to run it, **how** to fit it, and **when** to move data — then drives execution with **observability, resilience, and policy**.

It transforms model graphs (Torch, ONNX, TensorRT-LLM) into **validated execution plans**, applies quantisation to reduce memory footprint, **partitions** and **places** work across devices (as exported by Grendel), and coordinates the run to completion with **retries and checkpoints**.

Huginn exists to maximise **throughput and capacity** on commodity hardware without requiring invasive application rewrites. It is opinionated about **planning and reliability**, and intentionally light in the hot path: the heavy lifting stays inside Grendel and the frameworks themselves.

---

### What Huginn is

- A **planning engine** that converts high-level model graphs into **explicit plans** with resource budgets, memory movement, and kernel execution sequencing.
- An **orchestrator** that submits plans to Grendel, monitors execution, handles errors, and manages **sessions** (weights, KV cache, persistent buffers).
- A **quantisation manager** that integrates GPTQ, AWQ, and bitsandbytes-like schemes behind a common adapter, yielding smaller RAM/VRAM footprints.
- A **topology-aware placer** that uses NUMA/PCIe/NVLink hints to reduce data motion and balance loads.
- A **service boundary** with a small SDK/CLI: `huginn plan`, `huginn execute`, `huginn diagnose`, `huginn quantize`.

### What Huginn isn’t

- It is **not** a deep-learning framework; no new kernels or training loops live here.
- It is **not** a VRAM pooler — that’s outside v1 and counter to Grendel’s tiering model.
- It is **not** a cross-node scheduler; v1 focuses on a single host with multiple GPUs.
- It is **not** a replacement for Grendel; it **depends** on Grendel for enforcement, memory tiering, and “one device” illusion.

### Why it exists (problem statement)

- Large models easily exceed single-GPU VRAM. Even with Grendel’s tiering, **where** and **when** things move matters.
- Fragmented tooling makes **quantisation** and **placement** fragile and inconsistent across frameworks.
- Operators need **predictable, debuggable** plans: what is loaded, when transfers occur, which partials are cached, how retries work.
- Developers want **simple commands** that turn a model into a runnable plan and **get it done** on whatever cards they have.

### Positioning vs. Grendel & Muninn

- **Grendel** enforces and abstracts devices; **Huginn** decides placement and submits work. Grendel owns memory tiers and scheduling; Huginn provides **hints and constraints**.
- **Muninn** (optional) shares memory across GPUs/jobs. Huginn integrates by **requesting/renewing leases** and embedding those handles into plans. If Muninn isn’t present, Huginn **degrades gracefully**.

### Primary outcomes

1. A **reliable planner** that outputs readable, testable, schema-validated plans.
2. An **orchestrator** that can run long jobs with **resume** and **rollback**.
3. **Quantisation flows** that reduce memory 2–4× with measurable quality/latency tradeoffs.
4. **Topology-aware** placement that reduces interconnect traffic and stalls.
5. **Observability**: metrics, logs, and traces map cleanly from plan → execution.
6. **SDK/CLI** that app developers can use without learning GPU internals.

### Non-goals (v1)

- Training or fine-tuning in-core (future optional module).
- Tight VRAM pooling or inter-GPU unified memory (contra Grendel v1).
- Cross-node cluster scheduling (separate concern).
- Framework-specific kernel optimisations in Huginn core.

### Primary user personas

- **Infra/SRE:** wants predictable runs, clear metrics, and panic-free recovery.
- **ML platform dev:** wants a “make it fit and run” pipeline with quantisation choices.
- **Researcher/Indie:** wants to try big models on modest rigs without yak-shaving.

### Success criteria / KPIs

- **Plan quality:** ≥95% of plans validate without manual edits; plan realizations meet predicted memory budgets within ±10%.
- **Throughput:** Near-linear throughput scaling across independent jobs (benchmarks provided).
- **Footprint reduction:** Quantised runs reduce active VRAM by ≥40% with acceptable quality metrics (per backend).
- **Resilience:** Long jobs (≥8 hours) survive at least one orchestrated fault (simulated) without full restart.
- **Ops:** Clear per-job traces and Prometheus metrics; <150 µs overhead in Huginn hot path per submission/step scheduling event.

### Core constraints & assumptions

- Linux host(s), **NVIDIA first**; AMD/ROCm may be supportable via adapters later.
- **Python + Cython** implementation; C only for integration stubs if absolutely necessary.
- **camelCase** naming across internal code.
- **Huginn ⇄ Grendel** via REST/gRPC with stable protobuf/JSON schemas.

### Interfaces (at a glance)

- **Huginn CLI/SDK:** `diagnose`, `plan`, `execute`, `quantize`, `status`, `cancel`.
- **Huginn ⇄ Grendel:** `/devices`, `/plan`, `/execute`, `/jobs/{id}`, `/policy/kv`, `/weights/register`.
- **Huginn ⇄ Muninn:** `/lease`, `/renew`, `/release` (optional).
- **Planner artefacts:** `plan.json`, `topology.json`, `quantConfig.json`, `metrics/*.jsonl`.

### Main risks & mitigation

- **Plan–runtime drift:** enforce strict schema validation and runtime assertions; embed versions and hashes in plan headers.
- **Quantisation mismatch:** standardise adapters and add preflight tests that verify de/quant consistency on sample tensors.
- **Topology changes at runtime:** detect via Grendel; re-evaluate placement with safe, incremental re-planning.
- **User confusion:** provide annotated plan viewer and CLI `explain` commands; rich error messages.

### Acceptance signals

- Users can **diagnose → plan → execute** an ONNX/Torch model with default quantisation on a multi-GPU host **without manual edits**.
- A plan run yields **comparable** throughput to hand-managed runs and meets VRAM budgets on varied hardware.
- Fault injection suite demonstrates **resume** and **partial retry** without data corruption.

---

## 1. Guiding Principles

These principles define how contributors — whether AI or human — should think about design, implementation, and maintenance of Huginn.  
They are short, actionable, and intended to guide day-to-day decisions.

### 1. Planner-Centric Design

- **Plans before runs:** Huginn should never “wing it”. Every execution is the result of a **validated plan**.
- **Readable plans:** Human operators must be able to inspect, diff, and reason about `plan.json`.
- **Predictable budgets:** Plans carry explicit memory/latency estimates and **assertions** that the runtime enforces.

**Contributor note:** Prefer expanding the plan schema over sneaking new runtime flags. Treat the planner as the single source of truth.

### 2. Separation of Concerns

- Keep **planner**, **orchestrator**, **quantisation**, and **runtime adapters** isolated.
- Use **clean DTOs** (dataclasses) across boundaries. Avoid implicit shared state.
- Orchestrator is responsible for **control flow**, not heavy compute.

**Contributor note:** If a new feature spans modules, create a thin integration layer rather than deep coupling.

### 3. Quantisation-First Mindset

- Treat quantisation as a **first-class** feature, not a bolt-on script.
- Provide consistent **quality/perf knobs** (e.g., group size, calibration dataset) across GPTQ/AWQ/BnB.
- Plans capture quantisation parameters so runs are **reproducible**.

**Contributor note:** Every new backend must ship with a quantisation adapter + tests.

### 4. Observability and Debuggability

- Everything significant should be **measurable** and **traceable**: planner decisions, placement, retries, cache hits.
- Logs are **structured**; traces link plan nodes to runtime jobs; metrics are Prometheus-compatible.
- Provide a `huginn explain plan.json` command that narrates key choices.

**Contributor note:** New components must add metrics and traces in the same PR.

### 5. Security and Multi-Tenancy Awareness

- Assume multi-user environments: enforce **quotas**, respect ownership of weights and caches, and **scrub** sensitive data from logs.
- UDS permissions and optional mTLS for service mode; tokens for Huginn ⇄ Grendel calls.

**Contributor note:** Threat-model new endpoints; validate all inputs against schemas.

### 6. Topology-Aware Scheduling

- Prefer **locality** (NUMA node affinity, PCIe switch proximity); avoid congested links.
- Plans should tolerate topology change: define fallback placements and **cost deltas**.

**Contributor note:** Surface topology data in `topology.json`; make decisions explainable.

### 7. Minimal Hot Path Logic

- Runtime hot paths should be short and **boring**. Heavy heuristics belong in the **planner** or **pre-execution** phases.
- Avoid complex cross-threads signalling in hot paths; prefer queue-centric designs.

**Contributor note:** Profile before/after when touching any hot path; keep p99 latency stable.

### 8. Compatibility Without Lock-In

- Support Torch, ONNX, and TensorRT-LLM via adapters. Plans remain **framework-neutral**.
- Allow third parties to implement adapters without forking Huginn.

**Contributor note:** Keep adapter interfaces small and documented; avoid leaking internal types.

### 9. Determinism Where It Matters

- For the same inputs, planner should yield **stable** plans (within a tolerance band).
- Seed and record random choices (e.g., in calibration sampling) in the plan header.

**Contributor note:** Include determinism tests; gate changes that cause large plan diffs.

---

# 2. Milestones

The milestones define the staged delivery of Huginn from minimal skeleton to a production-ready planning and orchestration layer.  
Each milestone builds on the last, delivering working, testable increments.

---

## M0 — Core Package & CLI Skeleton

**Goal:** Establish the repository structure, CLI surface, config system, and basic wiring to existing modules (`aiAssistant`, `dataTypes`, `systemDiagnostics`).

### Features
- `huginn` Python package with CLI entrypoint (`huginn`).
- CLI commands: `diagnose`, `plan`, `execute`, `quantize`, `status`, `cancel`.
- Config loader:
  - TOML config with env var overrides.
  - JSON Schema validation.
  - Profiles (e.g., `dev`, `prod`) with inheritance.
- Logging subsystem: structured JSON, rotating file handler, CLI verbosity flags.
- Error taxonomy using the project's `noexcept` style for consistent reporting.
- `--dry-run` mode for all commands.

### Deliverables
- Running CLI with help text and examples.
- `config/defaults.toml` and `config/schema.json`.
- `docs/architecture.md` draft with component boundaries.
- `examples/` with a minimal end-to-end script (diagnose → plan → execute using a trivial model).
- `scripts/dev.ps1` and `scripts/dev.sh` to bootstrap the venv and run tests/lint/type-checks.

### Tests
- Unit: CLI parsing (`argparse`) and command handlers.
- Unit: Config loader validation with good/bad samples.
- Unit: Logging initialisation (levels, formats).
- Integration: `huginn diagnose` on a machine with/without CUDA.
- Integration: `huginn plan` with a mock planner returning a toy graph.
- Contract: Config schema round-trip (load → dump → load).

### Exit Criteria
- `huginn --help` lists the commands; each command runs in `--dry-run` without errors.
- On a non-CUDA machine, `diagnose` still returns a useful report.
- CI passes lint, type checks, and unit tests.

---

## M1 — Model Registry & System Diagnostics

**Goal:** Provide accurate hardware introspection and a first-pass model selection API to choose models that fit disk/RAM/VRAM constraints.

### Features
- `SystemDiagnostics` reports: total RAM, free RAM, GPU count, per-GPU VRAM, free disk space at cache path, CUDA/driver versions.
- Configurable cache root (HF_HOME et al.) with sanity checks and warnings.
- `dataTypes.py` defines `ModelSpec` (name, id, size, memory, tokenizers, context, tags) and a curated list of models.
- Model recommendation algorithm:
  - Inputs: diagnostics, quantisation preference, desired context length, latency/throughput preference.
  - Output: ordered list of `ModelSpec` with reasons.
- Tokeniser selection: prefer fast → fallback to slow; validation against installed transformers versions.
- `aiAssistant` updated to consume the registry and diagnostics (already exists — wire it cleanly).

### Deliverables
- `huginn diagnose` prints a table and JSON with hardware and recommended models.
- `huginn models list` and `huginn models show <id>` commands (read-only views).
- Documentation: how to add a model to the registry; policy for inclusion.
- Examples: `minimalTextGen.py` that uses the diagnostics + registry to load a model and say hello.

### Tests
- Unit: `SystemDiagnostics` on mocked platforms (CUDA present/absent, multiple GPUs, low disk).
- Unit: `ModelSpec` validation (tokenisers present; memory/size sanity).
- Unit: recommendation sorting (preferences respected; constraints enforced).
- Integration: `aiAssistant` happy-path loading with HF cache set to a temp dir (offline + online modes).
- Property-based tests: model selection monotonicity when resources increase.
- Doc tests: examples execute under `--dry-run` without downloading weights.

### Exit Criteria
- `huginn diagnose` suggests at least one viable model on a typical consumer GPU host.
- Model/tokeniser mismatches are caught early with useful error messages.
- Offline mode (`--local`) loads from cache if files exist; otherwise a clear prompt to provide a token or download.

---

## M2 — Planner Engine v1 (Graph & Cost)

**Goal:** Convert supported model formats into a **plan graph** with **cost estimates** and **resource budgets**, ready for placement and execution.

### Features
- **Graph ingestion**:
  - Torch FX/`torch._dynamo` export path for PyTorch models.
  - ONNX model loader with shape inference.
  - TensorRT-LLM graph reader (optional, as metadata).
- **IR (intermediate representation)** for plans: nodes (ops), edges (tensors), attributes (dtype, shape, size, lifetime).
- **Cost model**:
  - Memory: parameter weights, activations, KV cache growth per token, temporary buffers.
  - Compute: flops estimates per op; approximate latency per device class.
  - I/O: H2D/D2H transfer costs; NVMe spill/restore costs when Grendel tiers.
- **Constraints & hints**:
  - Hard constraints (must-run-on device type), soft hints (prefer device N), co-location groups, anti-affinity sets.
  - Quantisation annotations (allowed backends per subgraph).
- **Plan validation**:
  - Schema validation.
  - Budget checks (fits-in-device-window, fits-in-aggregate-host-RAM).
  - Determinism: stable node ordering; canonicalisation for diffing.
- **Visualisation**: export Graphviz/DOT and a JSON summary.

### Deliverables
- `huginn plan <modelPath|hubId>` produces:
  - `plan.json` (full IR with budgets and hints).
  - `plan.dot` (optional) and `plan.summary.json` (aggregates).
- `planner/validators.py` with explicit error messages (missing shapes, illegal cycles, over-budget nodes).
- `planner/costModel.py` with pluggable device profiles (e.g., 20-series, 30-series, 40-series GPUs).

### Tests
- Unit: IR builders (node/edge construction, lifetimes).
- Unit: cost model math (per-op flops; buffer sizes from shapes/dtypes).
- Unit: shape inference and failure modes.
- Integration: ingest a small Torch model, produce a plan, and validate budgets.
- Golden tests: snapshot `plan.json` for known toy models; detect regressions via diff.
- Fuzz: random small graphs to stress validators.

### Exit Criteria
- `plan.json` passes schema + budget checks for at least two sample models (Torch + ONNX).
- Cost estimates are within ±20% of measured latency/memory on reference hardware (documented calibration).
- Visualisation renders without errors for non-cyclic graphs.

---

## M3 — Orchestrator & Job Manager

**Goal:** Execute plans via Grendel, manage sessions (weights, KV cache, artifacts), and provide robust control flow with observability.

### Features
- **Execution engine**:
  - Translate plan nodes to Grendel API calls.
  - Stream-aware submission; overlap transfers/compute where possible (hints only; Grendel decides final scheduling).
  - Support for **checkpoints** (KV cache snapshots, progress markers).
- **Job manager**:
  - CRUD for jobs: submit, query status, cancel, pause/resume.
  - Priority queues; fair-share across users; quotas integration.
- **Sessions & artifacts**:
  - Weight registration; pinning in Grendel RAM or Muninn leases if available.
  - KV cache lifecycle management; eviction policy coordination.
  - Artifact store for logs/metrics/traces associated with a job.
- **Retries & recovery**:
  - Policy-based retries on transient failures.
  - Idempotent plan steps to allow partial re-execution.
- **Security**:
  - AuthN/Z for job operations; audit logs; per-user quotas.
- **CLI/SDK**:
  - `huginn execute plan.json` returns a job id.
  - `huginn status <jobId>` and `huginn cancel <jobId>`.

### Deliverables
- REST/gRPC server for orchestrator (service mode) + CLI client (local mode shares code paths).
- `orchestrator/executionEngine.py` with streaming submission and progress callbacks.
- `orchestrator/jobManager.py` with durable state (SQLite/postgres adapter; start with SQLite).
- `/jobs/{id}` schema with timestamps, states (QUEUED, RUNNING, PAUSED, FAILED, COMPLETED), metrics summary.
- `observability/metrics.py`: Prometheus metrics for jobs, steps, retries, and durations.
- End-to-end example: execute a small text generation plan; collect outputs and metrics.

### Tests
- Unit: state machine transitions and invalid transitions blocked.
- Unit: retry backoff logic and maximum attempts enforced.
- Integration: fake Grendel server to simulate device responses, including failures and timeouts.
- Integration: checkpoint/resume on a long-running toy workload.
- Contract: OpenAPI/protobuf generation and linting; SDK and server agree on schemas.
- Load: submit N jobs; verify fair-share and stable latency.

### Exit Criteria
- A plan produced by M2 runs end to end, yielding model output artifacts.
- Jobs survive a simulated Grendel hiccup (temporary /execute failure) and **resume**.
- `status` and metrics reflect accurate, timely state changes.

---

## M4 — Quantisation Integrations

**Goal:** Provide first-class quantisation flows to shrink models and activations, trading minimal accuracy for significant footprint reduction.

### Features
- **Backends**: GPTQ, AWQ, bitsandbytes (4/8-bit), with a consistent adapter interface.
- **Pipelines**:
  - Calibration data ingestion (text datasets; configurable sampling).
  - Parameter selection (which layers/weights to quantise).
  - Quality evaluation hooks (perplexity/probing tasks; plugin-friendly).
- **Runtime compatibility**:
  - Ensure quantised formats are consumable by the runtime adapters (Torch/ONNX/TRT-LLM) without surprises.
- **Plan annotations**:
  - Embed quantisation configs into `plan.json` for reproducibility.
- **Artifact management**:
  - Cache and version quantised weights; deduplicate with Grendel/Muninn when possible.

### Deliverables
- `quantization/` module with `loadQuantizer.py` and backend adapters (`gptq.py`, `awq.py`, `bitsandbytesAdapter.py`).
- CLI: `huginn quantize <model>` with flags for backend, group size, calibration size, etc.
- Benchmarks: speed/VRAM/quality across backends on small reference models (LLM8B, Gemma2B, etc.).
- Docs: quantisation trade-offs, knobs, and compatibility notes.

### Tests
- Unit: adapter argument validation and failure messages.
- Unit: calibration sampling (deterministic with seed).
- Integration: quantise a toy model and execute plan in M3 engine.
- Regression: ensure quantised weights load in supported backends.
- Quality gates: configurable thresholds per backend (e.g., Δperplexity ≤ X).

### Exit Criteria
- Quantised runs reduce active VRAM by ≥40% on reference hardware with acceptable quality delta.
- `plan.json` carries full quantisation metadata; runs are reproducible from artifacts.
- Benchmarks are reproducible and tracked over time.

---

## M5 — Topology-Aware Placement

**Goal:** Use NUMA/PCIe/NVLink awareness to **place** partitions where they run best, minimising data motion and congested links.

### Features
- **Topology discovery** via Grendel: device–NUMA mapping, PCIe switch layout, NVLink availability.
- **Placement policies**: locality-first, balanced, and throughput-max.
- **Plan rewriting**:
  - Insert transfer steps where needed; collapse steps where co-located.
  - Prefer co-location for chatty subgraphs; respect anti-affinity (e.g., thermal constraints).
- **Adaptive re-planning**:
  - Detect topology change at runtime (device drained, reset, throttled) and **shift** partitions safely.

### Deliverables
- `scheduling/topologyHints.py` and `scheduling/policy.py` with policy plug-ins.
- Metrics for interconnect bytes, transfer latencies, and placement decisions.
- Visualiser: highlight placement choices in `plan.dot` with colours per GPU/NUMA node.
- Docs: how to choose a policy; example outcomes per topology.

### Tests
- Unit: policy scoring (deterministic with same inputs).
- Integration: plan rewrite when moving from 1×GPU to 2×GPU.
- Integration: simulate a device removal during run; ensure fallback placement works.
- Benchmarks: reduced cross-switch transfers vs. naive placement (document targets).

### Exit Criteria
- On a dual-GPU host, topology-aware placement shows measurable improvement (e.g., ≥15% reduction in transfer time) vs. random placement on reference plans.
- Fallback placement occurs cleanly when a device is drained mid-run.

---

## M6 — Reliability, Recovery & Integration Hardening

**Goal:** Ensure Huginn can run **for days** with confidence: resilient to faults, transparent to operators, and locked-in on integration contracts.

### Features
- **Checkpoint/resume** at step and subgraph boundaries; restart after process or node reboot.
- **Audit trails**: immutable logs of job creation, changes, retries, cancellations, and outputs (hashed).
- **Contract tests**: versioned API schemas; deprecation warnings with timelines.
- **Operational tooling**:
  - `huginnctl` (optional) or extended CLI: drain, pause scheduling, snapshot state.
  - Health endpoints; readiness/liveness probes.
- **Backpressure and quotas**: prevent overload; coordinate with Grendel’s policies.

### Deliverables
- Replayable job journal (append-only) with compaction.
- Contract test suite that runs against a pinned Grendel build (and nightly against `main`).
- SRE runbook in `docs/operations.md` with failure modes and playbooks.
- Stress test harness (long-running synthetic workloads with fault injection).

### Tests
- Integration: power-cycle simulation for orchestrator process; verify resume from last checkpoint.
- Chaos: randomly kill worker processes; verify job-level guarantees.
- Contract: run all CLI commands against mocked and real Grendel endpoints; ensure no accidental breaking changes.
- Load: soak test (24h) with rotating jobs; memory/FD leaks tracked to zero.

### Exit Criteria
- 24-hour soak with zero fatal errors and successful job completions.
- All endpoints versioned and documented; deprecations flagged with metrics.
- SRE runbook validated via game-day drills.

---

## 3. Out of Scope (v1)

The following items are deliberately excluded from the v1 scope of Huginn.  
Contributors should avoid implementing or partially implementing these features unless explicitly approved.

### 1. Direct Model Training
- **Reason:** Huginn specialises in planning/execution for inference and light-weight adaptation workflows; training loops belong in frameworks.
- **Impact:** Focus keeps hot paths slim and avoids conflating roles with Grendel and frameworks.
- **Future Consideration:** Optional fine-tune orchestrators after v1, integrated with quantisation adapters.

### 2. Framework-Specific Optimisations in Core
- **Reason:** Tuning kernels or graph rewrites per framework leads to churn and tight coupling.
- **Impact:** Prevents lock-in and reduces maintenance.
- **Future Consideration:** Live in adapters and external plugins.

### 3. Tight VRAM Pooling
- **Reason:** Outside v1 design; conflicts with Grendel’s tiering model and reliability targets.
- **Impact:** Keeps memory semantics predictable.
- **Future Consideration:** Muninn fast-mode plus Grendel P2P in later releases.

### 4. Cross-Node Scheduling
- **Reason:** Adds distributed-systems complexity; v1 optimises single-host ROI first.
- **Impact:** Faster time-to-value with simpler ops.
- **Future Consideration:** Layer a cluster scheduler above Huginn once single-host is mature.

---

## 4. Contributor Guidelines

These guidelines apply to all contributors — human and AI — working on Huginn.

### 1. Code Style & Structure
- **Language:** Python + Cython; C only for minimal interop where unavoidable.
- **Naming:** camelCase for all functions and variables; respect external library casing.
- **Formatting:** Python `black` (120 columns), `isort` for imports; C/Cython via `clang-format` with repo rules.
- **Structure:** Maintain strict module boundaries; plan DTOs in `planner/`, orchestration in `orchestrator/`, adapters in `runtime/`.

**AI-specific note:** Always generate code that passes `make lint`, `make typecheck`, and `pytest -q` locally before proposing merge.

### 2. Commit & PR Practices
- **Commits:** Small, atomic; messages use `<component>: <summary>` (e.g., `planner: add activation buffer estimator`).
- **PRs:** Reference issues; include tests and docs updates; avoid mixing refactors with features.
- **Branching:** `main` stable; `feat/<area>/<short>` for features; `fix/<area>/<short>` for fixes.

### 3. Testing
- **Coverage:** 80% minimum for Python; Cython hot paths have dedicated unit tests.
- **Types of Tests:**
  - Unit tests for planners, estimators, and adapters.
  - Integration tests for CLI, planner → orchestrator → Grendel flow.
  - Regression “golden” tests for `plan.json` outputs.
  - Fault-injection tests for orchestrator recovery.
- **Tooling:** `pytest`, `pytest-asyncio`, coverage; CI matrices across Python/CUDA combos (as feasible).

### 4. Documentation
- Update `README.md`, `docs/architecture.md`, `docs/api.md`, and this roadmap when surface area changes.
- Provide runnable examples in `examples/` for new features.
- Keep change logs current; tag releases with semantic versions.

### 5. Performance Considerations
- Planner and orchestrator must not become the bottleneck; offload heavy work to background tasks.
- Cache expensive computations (e.g., topology digest, operator stats).
- Profile routinely; include p50/p90/p99 timings in PR descriptions for hot paths.

### 6. Observability & Logging
- Structured logs with correlation IDs (jobId, planNodeId).
- Prometheus metrics by default; optional OpenTelemetry tracing.
- Avoid excessive logging in tight loops; use debug flags.

### 7. Security & Safety
- Validate all external inputs (plan files, API requests) against schemas.
- Enforce per-user quotas where enabled; never expose raw secrets in logs.
- Use least-privilege defaults for filesystem and IPC.

### 8. Communication & Coordination
- Use issues for design proposals; summarise decisions in `AGENTS.md` or `docs/architecture.md`.
- Label issues consistently (`type/feature`, `priority/high`, etc.).
- Coordinate API changes with Grendel maintainers.

### 9. AI Contributor Guidance
- Be concise in code; expand in comments only when non-obvious.
- Provide reasons for design deviations.
- Prefer deterministic behaviour; document seeds and randomness explicitly.

### 10. Merging Rules
- No direct commits to `main` except hotfixes with maintainer approval.
- All other changes via PR with green CI and at least one reviewer approval.

---

## 5. Reference Documents

This section lists key documents to read, maintain, and reference while working on Huginn.

### 1. `AGENTS.md`
**Purpose:** Define responsibilities and handoffs across Huginn, Grendel, and Muninn.

**Contents:**
- Agent roles (Planner Agent, Orchestrator Agent, Quantisation Agent).
- Decision boundaries; conflict resolution.
- Standard operating procedures for releases and incidents.

**Usage:** Read before making changes that cross module boundaries.

### 2. `ARCHITECTURE.md`
**Purpose:** Explain Huginn’s internal design, with diagrams and flows.

**Contents:**
- Planner IR, cost model, and validators.
- Orchestrator state machine and persistence.
- Integration points with Grendel and (optionally) Muninn.
- Failure and recovery paths.

**Usage:** Consult before touching core logic.

### 3. `API.md`
**Purpose:** Document public APIs exposed by Huginn and its client SDK.

**Contents:**
- REST/gRPC endpoints; request/response schemas; error codes.
- CLI command semantics.
- Versioning and compatibility policy.

**Usage:** Reference when implementing or updating endpoints.

### 4. `GRENDEL_API.md`
**Purpose:** Describe Huginn ⇄ Grendel integration contracts.

**Contents:**
- `/devices`, `/plan`, `/execute`, `/jobs/{id}`, `/weights/register`.
- Placement hint formats and precedence.
- Error handling across boundaries.

**Usage:** Keep in lock-step with Grendel changes.

### 5. `ROADMAP.md` (this document)
**Purpose:** Track milestones, priorities, non-goals, and contributor rules.

**Usage:** Check before starting work to ensure alignment with current milestone.

### 6. External References
**Purpose:** Provide context on the tools and libraries used by Huginn.

**Contents:**
- PyTorch FX, ONNX IR, TensorRT-LLM docs.
- Quantisation libraries (GPTQ/AWQ/bitsandbytes).
- Prometheus and OpenTelemetry docs.

**Usage:** Ensure correct API usage and conceptual alignment.

### Maintenance Responsibility
- Each document has an owner listed in `AGENTS.md`.
- Outdated docs must be updated **in the same PR** that changes behaviour.
- CI will fail if certain modules are modified without touching their associated docs.

---

## Appendices

### A. Plan Schema Sketch (Non-Normative)
```json
{
  "version": "1.0",
  "createdAt": "2025-08-08T12:00:00Z",
  "model": {"id": "google/gemma-2b", "dtype": "fp16"},
  "topologyDigest": "…",
  "quant": {"backend": "gptq", "groupsize": 128, "seed": 42},
  "nodes": [
    {"id": "n1", "op": "embed", "deviceHint": "gpu:0", "bytes": 67108864},
    {"id": "n2", "op": "attn", "colocateWith": "n3", "bytes": 134217728}
  ],
  "edges": [
    {"from": "n1", "to": "n2", "dtype": "fp16", "shape": [1, 4096]}
  ],
  "budgets": {"vram": 12000000000, "host": 32000000000},
  "assertions": {"fits": true, "p95LatencyMs": 75}
}
```
**Note:** Actual schema lives in `planner/` and is validated via JSON Schema.

### B. Example CLI Workflows
1) **Quick start (offline):**
```
huginn diagnose --local
huginn plan google/gemma-2b --local --out plan.json
huginn execute plan.json --local
```
2) **Quantised run with calibration:**
```
huginn quantize meta-llama/Llama-3.1-8B --backend gptq --calib 512 --groupsize 128 --seed 1234
huginn plan meta-llama/Llama-3.1-8B --quant-config quant.json --out plan.json
huginn execute plan.json
```
3) **Topology-aware placement and resume:**
```
huginn plan ./models/my.onnx --policy locality-first --out plan.json
huginn execute plan.json
# After orchestrator restart
huginn status <jobId> --follow
```

### C. Metrics (Initial Set)
- `huginn_jobs_submitted_total`
- `huginn_jobs_running`
- `huginn_jobs_failed_total`
- `huginn_job_duration_seconds{quantized="true|false"}`
- `huginn_plan_nodes_total{backend="torch|onnx|trtllm"}`
- `huginn_bytes_transferred_total{direction="h2d|d2h"}`
- `huginn_retries_total{reason="xid|timeout|oom"}`
- `huginn_cache_hits_total{type="weights|kv"}`

### D. Error Codes (Sketch)
- `HGN-0001` InvalidConfig
- `HGN-0002` PlanSchemaViolation
- `HGN-0003` OverBudget
- `HGN-0004` IncompatibleQuantizer
- `HGN-0005` GrendelContractMismatch
- `HGN-0006` JobNotFound
- `HGN-0007` CheckpointMissing
- `HGN-0008` PermissionDenied