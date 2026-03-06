import {
  contractsMeta,
  mockTaskDetails,
  mockTasks,
  mockUploadedFiles,
} from "@/mock/data";
import type {
  ContractsMetaResponse,
  CreateTaskRequest,
  FilePurpose,
  TaskDetail,
  TaskQuery,
  TaskResultResponse,
  TaskStatusResponse,
  TaskSummary,
  UploadFileRequest,
  UploadFileResponse,
  UploadedFileMeta,
} from "@/types/contracts";

let tasksDb: TaskSummary[] = [...mockTasks];
let detailsDb: Record<string, TaskDetail> = structuredClone(mockTaskDetails);
let filesDb: UploadedFileMeta[] = [...mockUploadedFiles];

function delay(ms = 500): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function genId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function resolveDataFiles(payload: CreateTaskRequest): string[] {
  if (payload.mode === "static") {
    return [payload.inputs.jitterCsvFileId, payload.inputs.predictionCsvFileId].filter(Boolean) as string[];
  }
  return [payload.inputs.actualMotionCsvFileId, payload.inputs.idealMotionCsvFileId, payload.inputs.diffJitterCsvFileId].filter(
    Boolean
  ) as string[];
}

function fileNameById(id: string): string {
  return filesDb.find((f) => f.fileId === id)?.fileName ?? id;
}

function videoNameById(id: string): string {
  return filesDb.find((f) => f.fileId === id)?.fileName ?? "unknown-video.mp4";
}

function normalizePurpose(purpose: FilePurpose): string {
  return purpose.replace(/_/g, " ");
}

export async function uploadFile({ file, purpose }: UploadFileRequest): Promise<UploadFileResponse> {
  await delay(500);
  const id = genId("file");
  const response: UploadFileResponse = {
    fileId: id,
    fileName: file.name,
    fileType: file.type || "application/octet-stream",
    size: file.size,
    url: URL.createObjectURL(file),
    purpose,
    uploadedAt: new Date().toISOString(),
  };
  filesDb = [response, ...filesDb];
  return structuredClone(response);
}

export async function createTask(payload: CreateTaskRequest): Promise<TaskSummary> {
  await delay(600);
  const id = genId("task");
  const time = new Date().toISOString();
  const fileIds = resolveDataFiles(payload);
  const sourceVideoId = payload.inputs.sourceVideoFileId;
  const task: TaskSummary = {
    id,
    name: payload.name,
    description: payload.description,
    mode: payload.mode,
    status: "pending",
    createdAt: time,
    updatedAt: time,
    inputSummary: {
      videoName: videoNameById(sourceVideoId),
      dataFiles: fileIds.map(fileNameById),
    },
    resultSummary: {
      artifactCount: 0,
    },
  };

  const inputFiles = [sourceVideoId, ...fileIds].map((fileId) => filesDb.find((f) => f.fileId === fileId)).filter(Boolean) as UploadedFileMeta[];

  const detail: TaskDetail = {
    task: {
      ...task,
      params: payload.params,
      inputFiles,
    },
    inputOverview: {
      video: {
        durationSec: 0,
        fps: 0,
        resolution: "等待后端解析",
        codec: "-",
      },
      data: {
        samples: 0,
        dimensions: [],
        timeRangeSec: [0, 0],
        hasPredictionSegment: false,
        notes: `任务已创建。后端可根据文件 purpose 识别用途：${inputFiles.map((f) => normalizePurpose(f.purpose)).join(", ")}`,
      },
    },
  };

  tasksDb = [task, ...tasksDb];
  detailsDb[id] = detail;
  return structuredClone(task);
}

export async function getTasks(query?: TaskQuery): Promise<TaskSummary[]> {
  await delay(300);
  const { mode, status } = query ?? {};
  let list = [...tasksDb];
  if (mode) {
    list = list.filter((task) => task.mode === mode);
  }
  if (status) {
    list = list.filter((task) => task.status === status);
  }
  return structuredClone(list);
}

export async function getTaskDetail(taskId: string): Promise<TaskDetail> {
  await delay(350);
  const detail = detailsDb[taskId];
  if (!detail) {
    throw new Error("任务不存在");
  }
  return structuredClone(detail);
}

export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  await delay(250);
  const task = tasksDb.find((item) => item.id === taskId);
  if (!task) {
    throw new Error("任务不存在");
  }
  const progressMap: Record<TaskSummary["status"], number> = {
    pending: 10,
    processing: 60,
    completed: 100,
    failed: 100,
  };
  return {
    taskId,
    status: task.status,
    progress: progressMap[task.status],
    message:
      task.status === "completed"
        ? "任务已完成"
        : task.status === "failed"
          ? "处理失败，请检查输入数据格式"
          : task.status === "processing"
            ? "处理中：轨迹融合与稳像渲染"
            : "排队中",
    updatedAt: task.updatedAt,
  };
}

export async function getTaskResult(taskId: string): Promise<TaskResultResponse> {
  await delay(320);
  const detail = detailsDb[taskId];
  if (!detail) {
    throw new Error("任务不存在");
  }
  return {
    taskId,
    status: detail.task.status,
    result: detail.result,
  };
}

export async function downloadTaskArtifact(taskId: string, artifactName: string): Promise<string> {
  await delay(180);
  const detail = detailsDb[taskId];
  if (!detail?.result) {
    throw new Error("任务结果不存在");
  }
  const mapping: Record<string, string | undefined> = {
    stabilizedVideo: detail.result.artifacts.stabilizedVideoUrl,
    compareVideo: detail.result.artifacts.compareVideoUrl,
    keyframesActual: detail.result.artifacts.keyframesActualCsvUrl,
    keyframesResidual: detail.result.artifacts.keyframesResidualCsvUrl,
    metrics: detail.result.artifacts.metricsJsonUrl,
    report: detail.result.artifacts.reportPdfUrl ?? detail.result.artifacts.reportJsonUrl,
  };
  const url = mapping[artifactName];
  if (!url) {
    throw new Error("无效的 artifactName");
  }
  return url;
}

export async function getContractsMeta(): Promise<ContractsMetaResponse> {
  await delay(150);
  return structuredClone(contractsMeta);
}
