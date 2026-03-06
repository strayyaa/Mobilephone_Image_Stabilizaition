# 视频防抖展示与处理平台 前端实现详解

## 1. 文档目的

本文档用于完整说明当前前端阶段已完成的工作，重点包括：

1. 已实现的业务范围与边界
2. 技术架构与目录组织
3. 页面与交互设计细节
4. 组件拆分与复用策略
5. 数据类型与后端接口契约
6. Mock 数据与 Mock API 机制
7. 状态管理与数据流
8. 运行与构建结果
9. 后续对接与扩展建议

该文档可直接用于：

1. 比赛答辩讲解材料
2. 前后端联调交接文档
3. 后续团队开发基线

---

## 2. 当前版本定位

### 2.1 项目定位

当前项目是一个高保真前端展示系统，覆盖“输入 -> 处理 -> 结果 -> 导出”全流程交互，并通过完整类型定义预留后端接口。

### 2.2 当前明确不做

1. 不实现真实视频防抖算法推理
2. 不在浏览器内执行复杂模型计算
3. 不实现 Python 后端服务逻辑

### 2.3 当前明确已做

1. 双场景任务系统（静止场景、运动场景）
2. 完整任务生命周期页面（创建、列表、详情）
3. 高展示价值可视化（轨迹、残差、关键帧、指标、视频对比）
4. 开发者契约页（接口与文件格式说明）
5. Mock 数据演示链路可本地跑通
6. TypeScript 类型化接口契约，可直接给后端落地

---

## 3. 技术栈与工程基础

### 3.1 核心技术

1. React 18
2. TypeScript 5
3. Vite 5
4. Tailwind CSS 3
5. Radix primitives + shadcn 风格组件封装
6. Recharts（图表）
7. Framer Motion（页面/模块动效）
8. React Router 6（路由）
9. Zustand（状态管理）

### 3.2 工程配置文件

1. `package.json`：脚本与依赖
2. `vite.config.ts`：别名 `@` 指向 `src`
3. `tsconfig*.json`：TypeScript 构建与路径
4. `tailwind.config.ts`：深色主题变量、字体、阴影与背景扩展
5. `postcss.config.cjs`：Tailwind + Autoprefixer
6. `.eslintrc.cjs`：ESLint 规则

### 3.3 UI 风格实现要点

1. 深色背景 + 蓝紫点缀
2. 轻量 glassmorphism 卡片
3. 仪表盘式版面结构
4. 不夸张但有效的动效（卡片淡入、区块上浮）
5. 指标卡 + 图表 + 对比视频三者并重，适合答辩展示

---

## 4. 目录结构与职责划分

当前源码目录：

```text
src
├─ components
│  ├─ charts
│  │  └─ MotionChart.tsx
│  ├─ common
│  │  ├─ EmptyState.tsx
│  │  ├─ MetricCard.tsx
│  │  ├─ SectionHeader.tsx
│  │  └─ StatusBadge.tsx
│  ├─ detail
│  │  ├─ ResultArtifactsPanel.tsx
│  │  └─ VideoComparePanel.tsx
│  ├─ dev
│  │  └─ ApiContractTable.tsx
│  ├─ forms
│  │  └─ UploadZone.tsx
│  ├─ layout
│  │  ├─ AppLayout.tsx
│  │  ├─ Sidebar.tsx
│  │  └─ Topbar.tsx
│  ├─ tasks
│  │  └─ TaskCard.tsx
│  └─ ui
│     ├─ badge.tsx
│     ├─ button.tsx
│     ├─ card.tsx
│     ├─ input.tsx
│     ├─ separator.tsx
│     ├─ switch.tsx
│     ├─ tabs.tsx
│     └─ textarea.tsx
├─ features
│  ├─ tasks
│  │  └─ taskStore.ts
│  └─ uploads
│     └─ useFileUpload.ts
├─ hooks
│  └─ useInterval.ts
├─ lib
│  ├─ format.ts
│  └─ utils.ts
├─ mock
│  ├─ api.ts
│  └─ data.ts
├─ pages
│  ├─ DeveloperContractsPage.tsx
│  ├─ NewTaskPage.tsx
│  ├─ NotFoundPage.tsx
│  ├─ OverviewPage.tsx
│  ├─ TaskDetailPage.tsx
│  └─ TasksPage.tsx
├─ router
│  └─ index.tsx
├─ types
│  └─ contracts.ts
├─ App.tsx
├─ index.css
└─ main.tsx
```

职责分层说明：

1. `pages`：路由页面级业务编排
2. `components`：可复用视觉组件与业务模块
3. `features`：按业务域拆分状态与 hooks
4. `mock`：模拟接口和数据源
5. `types`：全局类型契约与 DTO
6. `lib/hooks`：工具函数与通用 Hook

---

## 5. 路由与页面实现说明

### 5.1 路由总览

1. `/`：总览页
2. `/tasks/new`：新建任务
3. `/tasks`：任务列表
4. `/tasks/:taskId`：任务详情
5. `/developer/contracts`：开发者契约页
6. `*`：404 页面

### 5.2 全局布局

统一布局由 `AppLayout` 提供：

1. 左侧固定导航 Sidebar
2. 顶部 Topbar（当前页面标题 + 时间 + 快捷入口）
3. 主内容区域 Outlet

该结构保证了仪表盘式体验和展示一致性。

### 5.3 总览页（OverviewPage）

已实现内容：

1. 顶部 Hero 区
2. 系统定位与简介文案
3. 模式入口卡片（静止/运动）
4. 核心指标卡（任务数、平均提升、处理中任务、产物类型）
5. 流程示意（输入上传、任务创建、后端处理、可视化、导出）
6. 系统亮点说明
7. 近期任务卡片

数据来源：

1. Zustand `taskStore.fetchTasks()`
2. 任务完成数据计算平均稳像提升比例

### 5.4 新建任务页（NewTaskPage）

页面采用 Tab 切换两种场景，并实现两个独立表单状态。

静止场景字段：

1. 任务名称
2. 任务说明
3. 原始视频上传
4. 抖动 CSV 上传
5. 预测 CSV 可选上传
6. 参数区：补偿强度、裁切策略、平滑级别、旋转补偿、输出分辨率策略

运动场景字段：

1. 任务名称
2. 任务说明
3. 原始视频上传
4. 实际运动 CSV 上传
5. 理想运动 CSV 上传
6. 差分抖动 CSV 可选上传
7. 参数区：自动差分、补偿强度、裁切策略、平滑级别、旋转补偿

交互能力：

1. 拖拽上传与点击上传
2. 上传后展示文件名、大小、状态与 fileId
3. 前端必填校验
4. 提交后调用 `createTaskAndRefresh`
5. 创建成功自动跳转任务详情页

### 5.5 任务列表页（TasksPage）

已实现：

1. 按状态筛选：全部/待处理/处理中/已完成/失败
2. 按模式筛选：全部/静止/运动
3. 任务卡片网格展示
4. 空状态与加载状态展示

任务卡片内容：

1. 任务名称
2. 场景类型
3. 状态徽标
4. 创建时间
5. 输入文件摘要
6. 详情跳转

### 5.6 任务详情页（TaskDetailPage）- 重点页面

模块 A 任务基本信息：

1. 任务名、ID、场景类型、状态
2. 创建时间/更新时间
3. 输入文件列表
4. 参数摘要

模块 B 输入数据概览：

1. 视频元信息（时长、帧率、分辨率、编码）
2. 数据样本点、维度、时间范围
3. prediction 段标记
4. 备注说明

模块 C 核心结果展示：

1. 稳像前后视频对比（左右双视频）
2. 抖动轨迹图（x/y/z）
3. 补偿后残差图
4. 平移与旋转关键帧图
5. 指标卡片（5 项）
6. 可解释输出说明
7. 导出区（视频、CSV、metrics、report）

模块 D 运动场景扩展：

1. 实际运动轨迹
2. 理想运动轨迹
3. 差分抖动轨迹
4. 差分说明文字

状态轮询：

1. 对处理中任务启用 `useInterval`
2. 定时调用 `/api/tasks/:id/status`
3. 展示进度消息

### 5.7 开发者契约页（DeveloperContractsPage）

该页面是前后端对接核心文档页面，已实现：

1. 接口列表表格（method/path/requestType/responseType）
2. 每个接口请求/响应示例 JSON
3. 输入文件格式约定（扩展名、建议字段、必填性）
4. 结果产物约定
5. static/dynamic 数据流步骤
6. Python 脚本接入参数与输出说明

数据来自：

1. `GET /api/meta/contracts` 的 mock 返回

---

## 6. 组件设计与复用策略

### 6.1 UI 基础组件（`components/ui`）

已封装：

1. Button
2. Card
3. Badge
4. Input
5. Textarea
6. Tabs
7. Switch
8. Separator

价值：

1. 页面风格一致
2. 简化业务页面代码
3. 后续替换为完整 shadcn 体系时迁移成本低

### 6.2 公共业务组件（`components/common`）

1. `SectionHeader`：区块标题 + 描述 + 操作
2. `StatusBadge`：任务状态可视化
3. `MetricCard`：指标卡展示
4. `EmptyState`：空数据占位

### 6.3 任务与详情组件

1. `TaskCard`：列表卡片
2. `UploadZone`：可拖拽上传区
3. `MotionChart`：轨迹图通用封装
4. `VideoComparePanel`：前后视频对比
5. `ResultArtifactsPanel`：结果导出区
6. `ApiContractTable`：契约表格

---

## 7. 数据类型与接口契约设计

类型文件：`src/types/contracts.ts`

### 7.1 关键领域模型

1. `SceneMode`：`static | dynamic`
2. `TaskStatus`：`pending | processing | completed | failed`
3. `UploadedFileMeta`：上传文件标准元信息
4. `TaskParams`：统一参数结构（兼容两场景）
5. `TaskSummary`：列表摘要
6. `TaskDetail`：详情聚合对象
7. `TaskResult`：图表 + 指标 + 产物 + 解释输出

### 7.2 创建任务请求体

1. `CreateStaticTaskRequest`
2. `CreateDynamicTaskRequest`
3. `CreateTaskRequest`（联合类型）

设计特点：

1. 明确 `mode` 区分
2. 输入文件统一通过 `fileId` 引用
3. 前端参数区完整保留，后端按需消费

### 7.3 契约元信息

1. `EndpointContract`
2. `FileFormatContract`
3. `ArtifactContract`
4. `DataFlowContract`
5. `PythonScriptContract`
6. `ContractsMetaResponse`

这些类型支撑开发者页面渲染与后续自动化文档生成。

---

## 8. Mock 数据与 Mock API 实现细节

### 8.1 Mock 数据（`src/mock/data.ts`）

包含：

1. 上传文件样例
2. 任务列表样例（含不同状态）
3. 任务详情样例（含 static 与 dynamic）
4. 图表曲线生成器（轨迹、残差、关键帧）
5. 接口契约元数据

特点：

1. 时间戳动态生成（更贴近真实演示）
2. 指标与图表在语义上自洽
3. 请求/响应示例直接来自类型结构

### 8.2 Mock API（`src/mock/api.ts`）

已模拟接口：

1. `uploadFile`
2. `createTask`
3. `getTasks`
4. `getTaskDetail`
5. `getTaskStatus`
6. `getTaskResult`
7. `downloadTaskArtifact`
8. `getContractsMeta`

实现机制：

1. 内存数据库（`tasksDb/detailsDb/filesDb`）
2. 统一 `delay` 模拟网络耗时
3. `structuredClone` 防止状态污染
4. 下载接口返回 url，页面用 `window.open` 触发下载行为

---

## 9. 状态管理与数据流

### 9.1 Zustand Store（`taskStore.ts`）

状态字段：

1. `tasks`
2. `selectedTask`
3. `loading`
4. `error`
5. `filters`

动作方法：

1. `fetchTasks`
2. `fetchTaskDetail`
3. `createTaskAndRefresh`
4. `setFilters`

### 9.2 上传状态 Hook（`useFileUpload.ts`）

1. 统一管理上传过程 loading/error
2. 上传成功返回 `UploadedFileMeta`
3. 便于后续替换真实上传 SDK

### 9.3 轮询 Hook（`useInterval.ts`）

1. 对处理中任务定时请求状态
2. 避免在页面中重复 setInterval 逻辑
3. 可复用到列表页批量状态刷新

---

## 10. 可视化与展示层实现

### 10.1 图表体系

统一使用 `MotionChart` + Recharts 组合：

1. CartesianGrid
2. 多折线 Legend
3. Tooltip 主题统一
4. 响应式容器

已覆盖图表类型：

1. 抖动轨迹 x/y/z
2. 残差曲线
3. 关键帧平移/旋转
4. 运动场景三轨迹对比

### 10.2 视频对比区域

`VideoComparePanel` 采用左右并排布局：

1. 左：处理前
2. 右：处理后
3. 空数据时展示占位状态

### 10.3 指标卡展示

`MetricCard` 用于关键 KPI：

1. 抖动幅度下降比例
2. 残差均方值
3. 最大位移
4. 平均旋转变化
5. 裁切稳定性评分

---

## 11. 后端对接就绪度说明

### 11.1 已就绪部分

1. 接口路径与方法统一
2. 请求体与响应体类型明确
3. 上传文件用途字段 `purpose` 明确
4. 任务详情聚合结构明确
5. 下载 artifact 约定明确
6. 合同元数据接口已形成

### 11.2 后端最小接入方案

后端只需做到：

1. 按 `contracts.ts` 返回 JSON 字段
2. 替换 `src/mock/api.ts` 对应函数为真实 HTTP 调用
3. 返回可访问的产物 URL
4. 对处理中任务提供状态进度

前端页面无需重构，可直接联调。

### 11.3 Python 脚本映射建议

脚本输入参数：

1. `--video`
2. `--csv`
3. `--outdir`
4. 可选：`--target_px --target_deg --zoom --sign_x --sign_y --sign_rot --interp`

脚本产物映射：

1. `video_after_pred_compensation.mp4` -> `stabilized_video_url`
2. `keyframes_actual.csv` -> `keyframes_actual.csv`
3. `keyframes_residual.csv` -> `keyframes_residual.csv`
4. 指标 JSON -> `metrics.json`
5. 报告 -> `report.json/report.pdf`

---

## 12. 运行与构建结果

已完成本地验证：

1. `npm install` 成功
2. `npm run build` 成功

构建提示：

1. 当前有 bundle size 警告（主要来自图表库）
2. 不影响本地运行与演示

---

## 13. 与需求清单对照

### 13.1 需求覆盖结论

用户提出的核心要求已覆盖：

1. 双模式任务创建
2. 多页面完整系统结构
3. 任务管理与状态筛选
4. 任务详情重点可视化
5. 视频对比与导出区
6. 开发者接口契约页
7. 全量 mock 数据与 mock API
8. 完整可运行工程配置

### 13.2 仍属未来扩展的部分

1. 真实后端服务实现
2. 真实视频处理和模型推理
3. 鉴权与权限体系
4. 自动化测试与 CI/CD

---

## 14. 下一步建议（若进入联调阶段）

1. 在 `src/lib` 增加 `httpClient.ts`（axios/fetch 封装）
2. 把 `mock/api.ts` 切为真实服务 API 层
3. 新增 OpenAPI 文档导出，与 `contracts.ts` 双向校验
4. 增加任务详情页的错误码展示与重试机制
5. 对大规模曲线数据增加抽样与分段加载优化

---

## 15. 文档与源码入口

建议优先阅读：

1. `README.md`（项目概览与启动）
2. `src/types/contracts.ts`（所有契约类型）
3. `src/mock/api.ts`（接口行为模拟）
4. `src/pages/TaskDetailPage.tsx`（核心展示页面）
5. `src/pages/DeveloperContractsPage.tsx`（对接文档页面）

本文件定位为“当前前端实现的完整交付说明”，后续每次迭代可按版本追加变更记录。

