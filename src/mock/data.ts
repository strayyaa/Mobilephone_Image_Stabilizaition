import type {
  ContractsMetaResponse,
  CreateDynamicTaskRequest,
  CreateStaticTaskRequest,
  EndpointContract,
  FileFormatContract,
  SeriesPoint,
  TaskDetail,
  TaskSummary,
  UploadedFileMeta,
} from "@/types/contracts";

const sourceVideoUrl = "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4";
const stabilizedVideoUrl = "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.webm";

const now = Date.now();

function atMinutesAgo(minutes: number): string {
  return new Date(now - minutes * 60 * 1000).toISOString();
}

function createSeries(samples: number, scale = 1): SeriesPoint[] {
  return Array.from({ length: samples }, (_, i) => {
    const t = Number((i * 0.04).toFixed(3));
    return {
      t,
      x: Number((Math.sin(i / 8) * scale + Math.sin(i / 17) * 0.3 * scale).toFixed(4)),
      y: Number((Math.cos(i / 12) * scale * 0.8).toFixed(4)),
      z: Number((Math.sin(i / 15) * scale * 0.55).toFixed(4)),
    };
  });
}

function createResidualSeries(samples: number): SeriesPoint[] {
  return Array.from({ length: samples }, (_, i) => ({
    t: Number((i * 0.04).toFixed(3)),
    residual: Number((Math.abs(Math.sin(i / 11)) * 0.16 + 0.02).toFixed(4)),
  }));
}

function createKeyframeSeries(samples: number): SeriesPoint[] {
  return Array.from({ length: samples }, (_, i) => ({
    t: i,
    translation: Number((Math.sin(i / 3.5) * 2.2 + 4.8).toFixed(3)),
    rotation: Number((Math.cos(i / 4.3) * 0.35 + 0.8).toFixed(3)),
  }));
}

export const mockUploadedFiles: UploadedFileMeta[] = [
  {
    fileId: "file_vid_static_01",
    fileName: "tripod_static_source.mp4",
    fileType: "video/mp4",
    size: 24_731_501,
    url: sourceVideoUrl,
    purpose: "source_video",
    uploadedAt: atMinutesAgo(90),
  },
  {
    fileId: "file_jitter_static_01",
    fileName: "tripod_jitter_trace.csv",
    fileType: "text/csv",
    size: 812_391,
    url: "/mock/keyframes_actual.csv",
    purpose: "jitter_csv",
    uploadedAt: atMinutesAgo(89),
  },
  {
    fileId: "file_pred_static_01",
    fileName: "tripod_prediction.csv",
    fileType: "text/csv",
    size: 703_102,
    url: "/mock/keyframes_residual.csv",
    purpose: "prediction_csv",
    uploadedAt: atMinutesAgo(89),
  },
  {
    fileId: "file_vid_dynamic_01",
    fileName: "drone_city_flight.mp4",
    fileType: "video/mp4",
    size: 87_312_090,
    url: sourceVideoUrl,
    purpose: "source_video",
    uploadedAt: atMinutesAgo(65),
  },
  {
    fileId: "file_actual_motion_01",
    fileName: "actual_motion_trace.csv",
    fileType: "text/csv",
    size: 1_420_144,
    url: "/mock/actual_motion.csv",
    purpose: "actual_motion_csv",
    uploadedAt: atMinutesAgo(64),
  },
  {
    fileId: "file_ideal_motion_01",
    fileName: "ideal_motion_trace.csv",
    fileType: "text/csv",
    size: 1_100_452,
    url: "/mock/ideal_motion.csv",
    purpose: "ideal_motion_csv",
    uploadedAt: atMinutesAgo(64),
  },
  {
    fileId: "file_diff_motion_01",
    fileName: "derived_jitter_diff.csv",
    fileType: "text/csv",
    size: 832_200,
    url: "/mock/diff_motion.csv",
    purpose: "diff_jitter_csv",
    uploadedAt: atMinutesAgo(63),
  },
];

export const mockTasks: TaskSummary[] = [
  {
    id: "task_static_001",
    name: "静止场景-手持轻抖动演示",
    description: "验证静止场景下基于历史抖动预测补偿的展示流程。",
    mode: "static",
    status: "completed",
    createdAt: atMinutesAgo(88),
    updatedAt: atMinutesAgo(42),
    inputSummary: {
      videoName: "tripod_static_source.mp4",
      dataFiles: ["tripod_jitter_trace.csv", "tripod_prediction.csv"],
    },
    resultSummary: {
      jitterReductionRatio: 0.734,
      residualMse: 0.0159,
      artifactCount: 5,
    },
  },
  {
    id: "task_dynamic_002",
    name: "运动场景-无人机轨迹差分",
    description: "实际运动与理想运动做差生成抖动数据并处理。",
    mode: "dynamic",
    status: "completed",
    createdAt: atMinutesAgo(67),
    updatedAt: atMinutesAgo(30),
    inputSummary: {
      videoName: "drone_city_flight.mp4",
      dataFiles: ["actual_motion_trace.csv", "ideal_motion_trace.csv", "derived_jitter_diff.csv"],
    },
    resultSummary: {
      jitterReductionRatio: 0.681,
      residualMse: 0.0202,
      artifactCount: 6,
    },
  },
  {
    id: "task_dynamic_003",
    name: "运动场景-教学样例任务",
    description: "展示自动差分开启后的处理中状态。",
    mode: "dynamic",
    status: "processing",
    createdAt: atMinutesAgo(20),
    updatedAt: atMinutesAgo(2),
    inputSummary: {
      videoName: "demo_uav_run.mp4",
      dataFiles: ["actual_motion_demo.csv", "ideal_motion_demo.csv"],
    },
    resultSummary: {
      artifactCount: 0,
    },
  },
  {
    id: "task_static_004",
    name: "静止场景-候选任务",
    mode: "static",
    status: "pending",
    createdAt: atMinutesAgo(9),
    updatedAt: atMinutesAgo(9),
    inputSummary: {
      videoName: "hallway_walk.mp4",
      dataFiles: ["hallway_jitter.csv"],
    },
  },
  {
    id: "task_static_005",
    name: "静止场景-异常示例",
    mode: "static",
    status: "failed",
    createdAt: atMinutesAgo(120),
    updatedAt: atMinutesAgo(119),
    inputSummary: {
      videoName: "broken_input.mp4",
      dataFiles: ["bad_trace.csv"],
    },
    resultSummary: {
      artifactCount: 0,
    },
  },
];

const staticDetail: TaskDetail = {
  task: {
    ...mockTasks[0],
    params: {
      compensationStrength: 78,
      cropStrategy: "adaptive",
      smoothingLevel: 7,
      enableRotationCompensation: true,
      outputResolutionStrategy: "smart_crop",
    },
    inputFiles: mockUploadedFiles.filter((f) =>
      ["file_vid_static_01", "file_jitter_static_01", "file_pred_static_01"].includes(f.fileId)
    ),
  },
  inputOverview: {
    video: {
      durationSec: 13.8,
      fps: 30,
      resolution: "1920x1080",
      codec: "H.264",
    },
    data: {
      samples: 414,
      dimensions: ["x_true", "y_true", "z_true", "x_pred", "y_pred", "z_pred", "phase"],
      timeRangeSec: [0, 13.8],
      hasPredictionSegment: true,
      notes: "phase 字段用于标识 history/prediction 两段，后端可按段统计。",
    },
  },
  result: {
    metrics: {
      jitterReductionRatio: 0.734,
      residualMse: 0.0159,
      maxDisplacementPx: 5.9,
      avgRotationChangeDeg: 0.78,
      cropStabilityScore: 0.93,
    },
    artifacts: {
      stabilizedVideoUrl,
      compareVideoUrl: sourceVideoUrl,
      keyframesActualCsvUrl: "/mock/keyframes_actual.csv",
      keyframesResidualCsvUrl: "/mock/keyframes_residual.csv",
      metricsJsonUrl: "/mock/metrics_static.json",
      reportJsonUrl: "/mock/report_static.json",
      reportPdfUrl: "/mock/report_static.pdf",
    },
    explainability: {
      compensationTrajectory: "补偿轨迹基于过去窗口预测值反向映射到每帧平移/旋转，形成 frame-wise 变换参数。",
      cropWindow: "裁切窗口采用 adaptive 策略，在稳像与视野保留之间做动态平衡，边缘留白阈值控制在 6%。",
      modelPlaceholder: "当前页面为推理结果占位展示，模型本体由离线脚本执行，前端仅消费 CSV 与指标文件。",
    },
    charts: {
      jitterTrajectory: createSeries(180, 1.15),
      residualSeries: createResidualSeries(180),
      keyframeSeries: createKeyframeSeries(40),
    },
  },
};

const dynamicDetail: TaskDetail = {
  task: {
    ...mockTasks[1],
    params: {
      autoComputeDiff: true,
      compensationStrength: 74,
      cropStrategy: "adaptive",
      smoothingLevel: 8,
      enableRotationCompensation: true,
      outputResolutionStrategy: "keep",
    },
    inputFiles: mockUploadedFiles.filter((f) =>
      ["file_vid_dynamic_01", "file_actual_motion_01", "file_ideal_motion_01", "file_diff_motion_01"].includes(
        f.fileId
      )
    ),
  },
  inputOverview: {
    video: {
      durationSec: 24.2,
      fps: 60,
      resolution: "3840x2160",
      codec: "H.265",
    },
    data: {
      samples: 1452,
      dimensions: ["timestamp", "x", "y", "z", "roll", "pitch", "yaw"],
      timeRangeSec: [0, 24.2],
      hasPredictionSegment: false,
      notes: "后端对实际轨迹与理想轨迹做时序对齐，再输出差分抖动用于补偿。",
    },
  },
  result: {
    metrics: {
      jitterReductionRatio: 0.681,
      residualMse: 0.0202,
      maxDisplacementPx: 8.6,
      avgRotationChangeDeg: 1.15,
      cropStabilityScore: 0.88,
    },
    artifacts: {
      stabilizedVideoUrl,
      compareVideoUrl: sourceVideoUrl,
      keyframesActualCsvUrl: "/mock/keyframes_actual_dynamic.csv",
      keyframesResidualCsvUrl: "/mock/keyframes_residual_dynamic.csv",
      metricsJsonUrl: "/mock/metrics_dynamic.json",
      reportJsonUrl: "/mock/report_dynamic.json",
      reportPdfUrl: "/mock/report_dynamic.pdf",
    },
    explainability: {
      compensationTrajectory: "差分得到的抖动轨迹经过平滑后反向作用到渲染轨道，生成最终稳定输出。",
      cropWindow: "运动场景中裁切窗口允许更高缩放弹性，避免高速转向导致黑边。",
      modelPlaceholder: "后续后端可接入 Python 脚本输出 keyframes_actual/residual.csv 与处理后视频。",
    },
    charts: {
      jitterTrajectory: createSeries(180, 1.25),
      residualSeries: createResidualSeries(180),
      keyframeSeries: createKeyframeSeries(40),
      actualMotionTrajectory: createSeries(180, 1.45),
      idealMotionTrajectory: createSeries(180, 1.1),
      diffTrajectory: createSeries(180, 0.72),
    },
  },
};

export const mockTaskDetails: Record<string, TaskDetail> = {
  task_static_001: staticDetail,
  task_dynamic_002: dynamicDetail,
  task_dynamic_003: {
    task: {
      ...mockTasks[2],
      params: {
        autoComputeDiff: true,
        compensationStrength: 70,
        cropStrategy: "adaptive",
        smoothingLevel: 6,
        enableRotationCompensation: true,
        outputResolutionStrategy: "keep",
      },
      inputFiles: [],
    },
    inputOverview: {
      video: {
        durationSec: 21.1,
        fps: 30,
        resolution: "1920x1080",
        codec: "H.264",
      },
      data: {
        samples: 0,
        dimensions: ["timestamp", "x", "y", "z"],
        timeRangeSec: [0, 0],
        hasPredictionSegment: false,
        notes: "处理中任务暂无完整数据摘要。",
      },
    },
  },
  task_static_004: {
    task: {
      ...mockTasks[3],
      params: {
        compensationStrength: 60,
        cropStrategy: "fixed",
        smoothingLevel: 5,
        enableRotationCompensation: false,
        outputResolutionStrategy: "fixed_1080p",
      },
      inputFiles: [],
    },
    inputOverview: {
      video: {
        durationSec: 0,
        fps: 0,
        resolution: "-",
        codec: "-",
      },
      data: {
        samples: 0,
        dimensions: [],
        timeRangeSec: [0, 0],
        hasPredictionSegment: false,
        notes: "待处理任务等待后端消费输入文件。",
      },
    },
  },
  task_static_005: {
    task: {
      ...mockTasks[4],
      params: {
        compensationStrength: 65,
        cropStrategy: "adaptive",
        smoothingLevel: 6,
        enableRotationCompensation: true,
        outputResolutionStrategy: "keep",
      },
      inputFiles: [],
    },
    inputOverview: {
      video: {
        durationSec: 7.1,
        fps: 25,
        resolution: "1280x720",
        codec: "H.264",
      },
      data: {
        samples: 180,
        dimensions: ["timestamp", "x_true", "y_true"],
        timeRangeSec: [0, 7.1],
        hasPredictionSegment: false,
        notes: "CSV 缺少 z 轴字段导致任务失败。",
      },
    },
  },
};

const createTaskReqStaticExample: CreateStaticTaskRequest = {
  mode: "static",
  name: "演示任务-静止场景",
  description: "上传原始视频+抖动CSV进行稳像展示",
  inputs: {
    sourceVideoFileId: "file_vid_static_01",
    jitterCsvFileId: "file_jitter_static_01",
    predictionCsvFileId: "file_pred_static_01",
  },
  params: {
    compensationStrength: 75,
    cropStrategy: "adaptive",
    smoothingLevel: 8,
    enableRotationCompensation: true,
    outputResolutionStrategy: "smart_crop",
  },
};

const createTaskReqDynamicExample: CreateDynamicTaskRequest = {
  mode: "dynamic",
  name: "演示任务-运动场景",
  description: "上传实际与理想轨迹，后端自动差分",
  inputs: {
    sourceVideoFileId: "file_vid_dynamic_01",
    actualMotionCsvFileId: "file_actual_motion_01",
    idealMotionCsvFileId: "file_ideal_motion_01",
    diffJitterCsvFileId: "file_diff_motion_01",
  },
  params: {
    autoComputeDiff: true,
    compensationStrength: 72,
    cropStrategy: "adaptive",
    smoothingLevel: 7,
    enableRotationCompensation: true,
    outputResolutionStrategy: "keep",
  },
};

const taskDetailExample = {
  task: {
    ...staticDetail.task,
    inputFiles: staticDetail.task.inputFiles.slice(0, 2),
  },
  inputOverview: staticDetail.inputOverview,
  result: {
    metrics: staticDetail.result?.metrics,
    artifacts: staticDetail.result?.artifacts,
    explainability: staticDetail.result?.explainability,
  },
};

const endpointContracts: EndpointContract[] = [
  {
    name: "创建任务",
    method: "POST",
    path: "/api/tasks",
    description: "支持 static / dynamic 两种任务体，返回任务摘要。",
    requestType: "CreateTaskRequest",
    responseType: "TaskSummary",
    requestExample: [createTaskReqStaticExample, createTaskReqDynamicExample],
    responseExample: mockTasks[0],
  },
  {
    name: "上传文件",
    method: "POST",
    path: "/api/files/upload",
    description: "上传输入或结果文件，返回 fileId 与可访问 URL。",
    requestType: "multipart/form-data + purpose",
    responseType: "UploadFileResponse",
    responseExample: mockUploadedFiles[0],
  },
  {
    name: "获取任务列表",
    method: "GET",
    path: "/api/tasks?status=&mode=",
    description: "支持按状态、场景过滤。",
    requestType: "TaskQuery",
    responseType: "TaskSummary[]",
    responseExample: mockTasks.slice(0, 2),
  },
  {
    name: "获取任务详情",
    method: "GET",
    path: "/api/tasks/:id",
    description: "返回完整输入摘要、图表数据、指标与导出链接。",
    requestType: "path:id",
    responseType: "TaskDetail",
    responseExample: taskDetailExample,
  },
  {
    name: "获取任务状态",
    method: "GET",
    path: "/api/tasks/:id/status",
    description: "用于轮询任务处理状态。",
    requestType: "path:id",
    responseType: "TaskStatusResponse",
    responseExample: {
      taskId: "task_dynamic_003",
      status: "processing",
      progress: 62,
      message: "轨迹差分阶段进行中",
      updatedAt: atMinutesAgo(1),
    },
  },
  {
    name: "获取任务结果",
    method: "GET",
    path: "/api/tasks/:id/result",
    description: "返回任务结果聚合对象，未完成时 result 为空。",
    requestType: "path:id",
    responseType: "TaskResultResponse",
    responseExample: {
      taskId: "task_static_001",
      status: "completed",
      result: staticDetail.result,
    },
  },
  {
    name: "下载结果文件",
    method: "GET",
    path: "/api/tasks/:id/download/:artifactName",
    description: "artifactName 可为 stabilizedVideo/keyframesActual/keyframesResidual/metrics/report。",
    requestType: "path:id + artifactName",
    responseType: "302 redirect or binary stream",
    responseExample: {
      artifactName: "keyframesActual",
      url: "/mock/keyframes_actual.csv",
    },
  },
  {
    name: "获取接口文档元信息",
    method: "GET",
    path: "/api/meta/contracts",
    description: "返回接口、数据格式、脚本接入说明，供前端契约页渲染。",
    requestType: "none",
    responseType: "ContractsMetaResponse",
  },
];

const fileFormats: FileFormatContract[] = [
  {
    name: "原始视频文件",
    scene: "both",
    required: true,
    extensions: [".mp4", ".mov", ".avi"],
    columns: [],
    description: "用于生成稳像前后对比，后端保持原始帧率与时间戳。",
  },
  {
    name: "静止场景抖动数据",
    scene: "static",
    required: true,
    extensions: [".csv"],
    columns: ["timestamp", "x_true", "y_true", "z_true", "x_pred?", "y_pred?", "z_pred?", "phase?"],
    description: "phase 可区分 history/prediction 段。",
  },
  {
    name: "运动场景实际运动数据",
    scene: "dynamic",
    required: true,
    extensions: [".csv"],
    columns: ["timestamp", "x", "y", "z", "roll?", "pitch?", "yaw?"],
    description: "后端做时序对齐。",
  },
  {
    name: "运动场景理想运动数据",
    scene: "dynamic",
    required: true,
    extensions: [".csv"],
    columns: ["timestamp", "x", "y", "z", "roll?", "pitch?", "yaw?"],
    description: "与实际轨迹做差分，得到抖动轨迹。",
  },
  {
    name: "差分抖动数据",
    scene: "dynamic",
    required: false,
    extensions: [".csv"],
    columns: ["timestamp", "dx", "dy", "dz", "droll?", "dpitch?", "dyaw?"],
    description: "若上传该文件，可跳过自动差分阶段。",
  },
];

export const contractsMeta: ContractsMetaResponse = {
  version: "v1.0.0",
  generatedAt: new Date(now).toISOString(),
  endpoints: endpointContracts,
  fileFormats,
  artifacts: [
    { name: "stabilized_video_url", required: true, description: "稳像后视频链接" },
    { name: "compare_video_url", required: false, description: "拼接对比视频链接（可选）" },
    { name: "keyframes_actual.csv", required: true, description: "原始轨迹关键帧参数" },
    { name: "keyframes_residual.csv", required: true, description: "补偿后残差关键帧参数" },
    { name: "metrics.json", required: true, description: "核心指标汇总" },
    { name: "report.json / report.pdf", required: false, description: "报告产物（可选）" },
  ],
  flows: [
    {
      scene: "static",
      steps: [
        "上传原始视频",
        "上传抖动CSV（可选预测段）",
        "调用 /api/tasks 创建任务",
        "轮询 /status 或直接查询 /result",
        "展示轨迹、残差、关键帧与导出链接",
      ],
    },
    {
      scene: "dynamic",
      steps: [
        "上传原始视频",
        "上传实际运动CSV + 理想运动CSV",
        "后端对齐并差分（或使用已上传差分CSV）",
        "进入稳像处理与结果生成",
        "展示实际/理想/差分轨迹与稳像结果",
      ],
    },
  ],
  pythonIntegration: {
    command: "python stabilize.py --video <video_path> --csv <trace_csv> --outdir <output_dir>",
    requiredArgs: ["--video", "--csv", "--outdir"],
    optionalArgs: ["--target_px", "--target_deg", "--zoom", "--sign_x", "--sign_y", "--sign_rot", "--interp"],
    outputs: [
      "video_actual_jitter.mp4",
      "video_after_pred_compensation.mp4",
      "keyframes_actual.csv",
      "keyframes_residual.csv",
    ],
  },
};
