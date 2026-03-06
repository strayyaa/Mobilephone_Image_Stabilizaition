# 视频防抖展示与处理平台（前端 + 接口契约）

一个基于 `React + TypeScript + Vite` 的本地可运行系统，用于展示“基于过去抖动信息预测未来抖动并反向补偿”的工程流程。  
当前版本**不实现真实算法推理**，重点提供：

- 高质量前端展示与交互流程
- 静止场景 / 运动场景双模式任务创建
- 任务列表与任务详情可视化
- 轨迹/残差/关键帧图表展示
- 结果导出入口（mock）
- 面向后端开发者的完整接口契约页

## 1. 项目简介

该系统定位为比赛答辩与工程对接平台，强调闭环流程：

`输入上传 -> 任务发起 -> 处理状态 -> 结果可视化 -> 导出产物`

支持两类场景：

- `static`（静止场景）：上传原始视频 + 抖动 CSV（可选 prediction CSV）
- `dynamic`（运动场景）：上传原始视频 + 实际运动 CSV + 理想运动 CSV（可选差分 CSV）

## 2. 本地启动方式

```bash
npm install
npm run dev
```

默认本地地址：

- `http://localhost:5173`

生产构建：

```bash
npm run build
npm run preview
```

## 3. 页面说明

- `/` 总览页
  - 系统简介、亮点、模式入口、流程示意、指标卡片、近期任务
- `/tasks/new` 新建任务页
  - `静止场景` / `运动场景` 双标签表单
  - 拖拽上传、参数配置、字段校验、任务提交
- `/tasks` 任务列表页
  - 按状态与场景过滤，卡片展示任务摘要
- `/tasks/:taskId` 任务详情页（重点）
  - A 基本信息
  - B 输入概览
  - C 核心结果（视频对比、轨迹图、残差图、关键帧图、指标卡、可解释输出、导出区）
  - D 运动场景扩展（实际轨迹、理想轨迹、差分图）
- `/developer/contracts` 开发者契约页
  - 接口列表、请求响应示例、文件格式、结果产物、数据流、Python 脚本接入说明

## 4. mock 说明

项目使用 `src/mock` 提供完整 mock 数据和 mock API：

- `src/mock/data.ts`
  - 任务列表 mock
  - 任务详情 mock
  - 抖动/残差/关键帧图表 mock
  - 接口契约元信息 mock
- `src/mock/api.ts`
  - 模拟异步接口（包含延迟）
  - 上传、创建任务、列表、详情、状态、结果、下载、契约元信息

> 当前上传行为会在前端生成 `fileId` 与 `URL.createObjectURL`，用于本地演示。

## 5. 后端对接方式

前端类型契约定义于：

- `src/types/contracts.ts`

核心接口（已实现 mock）：

1. `POST /api/tasks`
2. `POST /api/files/upload`
3. `GET /api/tasks`
4. `GET /api/tasks/:id`
5. `GET /api/tasks/:id/status`
6. `GET /api/tasks/:id/result`
7. `GET /api/tasks/:id/download/:artifactName`
8. `GET /api/meta/contracts`

后端只需按上述字段返回，即可直接替换 `src/mock/api.ts` 的实现。

## 6. 接口与文件格式约定

详见页面：

- `/developer/contracts`

覆盖内容：

- static / dynamic 两种任务请求体
- 上传文件用途字段（`FilePurpose`）
- 输入 CSV 推荐列定义
- 结果文件产物约定
- 脚本接入参数与产物映射

## 7. 现有 Python 脚本接入约定（已在开发者页体现）

后端未来可调用：

```bash
python stabilize.py --video <video_path> --csv <trace_csv> --outdir <output_dir> \
  [--target_px ... --target_deg ... --zoom ... --sign_x ... --sign_y ... --sign_rot ... --interp ...]
```

脚本输出建议映射到接口产物：

- `video_actual_jitter.mp4`
- `video_after_pred_compensation.mp4`
- `keyframes_actual.csv`
- `keyframes_residual.csv`

## 8. 目录结构

```text
src
├─ components
│  ├─ charts
│  ├─ common
│  ├─ detail
│  ├─ dev
│  ├─ forms
│  ├─ layout
│  ├─ tasks
│  └─ ui
├─ features
│  ├─ tasks
│  └─ uploads
├─ hooks
├─ lib
├─ mock
├─ pages
├─ router
└─ types
```

## 9. 后续扩展建议

1. 接入真实后端（FastAPI/Flask）并替换 mock API。
2. 任务详情增加实时轮询进度条与 WebSocket 推送。
3. 引入鉴权、租户隔离与下载签名 URL。
4. 增加报告模板导出（PDF）和实验版本管理。
5. 对大数据量曲线增加抽样/分段渲染优化。
