# SatNav Frontend / 前端

[中文](#中文) | [English](#english)

---

# English

SwiftVLN **real-world navigation console** (React 18 + TypeScript + Vite).

The UI is **operator-stepped**: inference and flight execution are decoupled. The frontend owns pacing, action-queue state, and button gating. There is no server-side auto-flight loop yet.

## Design mockup

See [`design/ui-mockup.png`](design/ui-mockup.png).

## Layering

```text
src/api/     Call side — REST clients, types, workflows (decoupled from UI)
src/ui/      Display side — pages, components, hooks
```

The UI talks to the backend **only** through `@/api` (no direct `fetch` in components).

## Development

```bash
cd satnav/frontend
npm install
npm run dev
```

- Page: `http://127.0.0.1:5173`
- API base: `http://127.0.0.1:8000` (`VITE_API_BASE` in `.env.development`; see `.env.example`)
- Backend must set `SATNAV_CORS_ORIGINS=http://127.0.0.1:5173` (match the page Origin; prefer `127.0.0.1` over mixing with `localhost`)

## Build

```bash
npm run build
npm run preview
```

## Console layout

```text
[Header: health badges + Session + settings]
[Instruction + Infer]     [Flight: bind / login / DRC + OSD]
[RTMP raw] [448×448]      [Action ×4 + Execute / Emergency STOP]
[Logs]
```

## Operator loop (frontend)

```text
register_device → login → drc
  → inference (actions / next_action / remaining_actions)
  → action queue records current slot (1–4)
  → execute one step: forward or turn (task_id)
  → poll stick-task until COMPLETED / FAILED
  → inference again (feedback frame, advance currentIndex)
  → loop until next_action == 0 or emergency STOP
```

Rules (implemented in `useConsoleController` + `actionQueue.ts`):

- **Inference** only calls `POST /api/satnav/model/inference`; **execute one step** only calls `forward` / `turn`.
- Do not infer again while flight is incomplete (`canRunInference` gate).
- First frame with `performed_inference=true` fills 1–4 slots; feedback frames have `actions=[]`.
- After STOP (`next_action=0` or emergency STOP), re-acquire DRC; inference stays available for debug.

## Buttons & API mapping

| UI (Chinese label) | API / behavior |
|--------------------|----------------|
| 摇控/无人机绑定 | `POST .../register_device` |
| 登录飞控系统 | `POST .../login` |
| 获取飞行控制 | `POST .../drc` |
| 推理 | `POST .../model/inference` only |
| 执行一步 | `POST .../flight/forward` or `.../flight/turn` (no inference) |
| stick-task refresh (icon) | `GET .../flight/stick-task/{task_id}` |
| ■ 应急 STOP | UI state only (no backend STOP flight API); may defer if stick-task running |
| 刷新 RTMP 服务 (header ⚙) | `POST .../system/rtmp/refresh` |

OSD pose is polled from `GET .../flight/osd/latest` while DRC is ready.

Login / device defaults are empty in source; fill in the UI or use local notes (`satnav_readme.local.md`, gitignored).

## Directory

```text
frontend/
├── design/ui-mockup.png
├── src/api/
│   ├── clients/       # one function per REST endpoint
│   ├── workflows/     # multi-step orchestration (e.g. flightSetup, stickPoll)
│   └── types/         # aligned with backend/API_DOCS.md
└── src/ui/
    ├── components/
    ├── hooks/         # useConsoleController (main state machine)
    ├── pages/
    └── utils/         # actionQueue, stickProgress
```

## Related docs

- Ground-station overview: [`../README.md`](../README.md)
- REST contracts: [`../backend/API_DOCS.md`](../backend/API_DOCS.md)

---

# 中文

SwiftVLN **实机导航控制台**（React 18 + TypeScript + Vite）。

当前为**操作员分步**模式：推理与飞控执行解耦，由前端控制节奏、维护 action 队列与按钮门控；服务端自动飞控闭环尚未实现。

## 设计稿

见 [`design/ui-mockup.png`](design/ui-mockup.png)。

## 分层

```text
src/api/     调用侧（REST 封装、类型、workflow）— 与 UI 解耦
src/ui/      显示侧（页面、组件、hooks）
```

UI **只**通过 `@/api` 访问后端，不在组件里直接 `fetch`。

## 开发

```bash
cd satnav/frontend
npm install
npm run dev
```

- 页面：`http://127.0.0.1:5173`
- API：`http://127.0.0.1:8000`（`.env.development` 中 `VITE_API_BASE`；参考 `.env.example`）
- 后端需配置 `SATNAV_CORS_ORIGINS=http://127.0.0.1:5173`（与页面 Origin 一致；避免 `localhost` 与 `127.0.0.1` 混用）

## 构建

```bash
npm run build
npm run preview
```

## 控制台布局

```text
[Header：健康徽章 + Session + 设置]
[导航指令 + 推理]         [飞控：绑定 / 登录 / DRC + OSD]
[RTMP 原图] [448×448]     [Action ×4 + 执行一步 / 应急 STOP]
[日志]
```

## 操作员主循环（前端）

```text
register_device → login → drc
  → inference（actions / next_action / remaining_actions）
  → action 队列记录当前槽位（1–4）
  → 执行一步：forward 或 turn（得 task_id）
  → 轮询 stick-task 至 COMPLETED / FAILED
  → 再次 inference（反馈帧，推进 currentIndex）
  → 循环直至 next_action == 0 或应急 STOP
```

规则（`useConsoleController` + `actionQueue.ts`）：

- **推理**仅调 `POST /api/satnav/model/inference`；**执行一步**仅调 `forward` / `turn`。
- 飞控未完成前禁止再次推理（`canRunInference` 门控）。
- 首帧 `performed_inference=true` 填充 1–4 个槽位；反馈帧 `actions=[]`。
- STOP（`next_action=0` 或应急 STOP）后需重新获取飞行控制；推理仍可用于调试。

## 按钮说明（与后端对齐）

| UI | API / 行为 |
|----|------------|
| 摇控/无人机绑定 | `POST .../register_device` |
| 登录飞控系统 | `POST .../login` |
| 获取飞行控制 | `POST .../drc` |
| 推理 | 仅 `POST .../model/inference` |
| 执行一步 | `POST .../flight/forward` 或 `.../flight/turn`（不触发推理） |
| stick-task 刷新（图标） | `GET .../flight/stick-task/{task_id}` |
| ■ 应急 STOP | 仅 UI 状态（后端无 STOP 飞控接口）；飞控执行中可能延迟生效 |
| 刷新 RTMP 服务（Header ⚙） | `POST .../system/rtmp/refresh` |

DRC 就绪后轮询 `GET .../flight/osd/latest` 展示 OSD。

登录 / 设备默认值在源码中为空；请在 UI 填写或查阅本地笔记（`satnav_readme.local.md`，已 gitignore）。

## 目录

```text
frontend/
├── design/ui-mockup.png
├── src/api/
│   ├── clients/       # 一接口一函数
│   ├── workflows/     # 分步闭环编排
│   └── types/         # 与 API_DOCS 对齐
└── src/ui/
    ├── components/
    ├── hooks/         # useConsoleController（主状态机）
    ├── pages/
    └── utils/         # actionQueue、stickProgress
```

## 相关文档

- 地面站总览：[`../README.md`](../README.md)
- REST 契约：[`../backend/API_DOCS.md`](../backend/API_DOCS.md)
