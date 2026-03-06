export type SceneMode = "static" | "dynamic";

export type TaskStatus = "pending" | "processing" | "completed" | "failed";

export type CropStrategy = "adaptive" | "fixed" | "none";
export type OutputResolutionStrategy = "keep" | "smart_crop" | "fixed_1080p";

export type FilePurpose =
  | "source_video"
  | "jitter_csv"
  | "prediction_csv"
  | "actual_motion_csv"
  | "ideal_motion_csv"
  | "diff_jitter_csv"
  | "artifact";

export interface UploadedFileMeta {
  fileId: string;
  fileName: string;
  fileType: string;
  size: number;
  url: string;
  purpose: FilePurpose;
  uploadedAt: string;
}

export interface TaskParams {
  compensationStrength: number;
  cropStrategy: CropStrategy;
  smoothingLevel: number;
  enableRotationCompensation: boolean;
  outputResolutionStrategy?: OutputResolutionStrategy;
  autoComputeDiff?: boolean;
}

export interface BaseCreateTaskRequest {
  name: string;
  description?: string;
  mode: SceneMode;
  params: TaskParams;
}

export interface CreateStaticTaskRequest extends BaseCreateTaskRequest {
  mode: "static";
  inputs: {
    sourceVideoFileId: string;
    jitterCsvFileId: string;
    predictionCsvFileId?: string;
  };
}

export interface CreateDynamicTaskRequest extends BaseCreateTaskRequest {
  mode: "dynamic";
  inputs: {
    sourceVideoFileId: string;
    actualMotionCsvFileId: string;
    idealMotionCsvFileId: string;
    diffJitterCsvFileId?: string;
  };
}

export type CreateTaskRequest = CreateStaticTaskRequest | CreateDynamicTaskRequest;

export interface TaskSummary {
  id: string;
  name: string;
  description?: string;
  mode: SceneMode;
  status: TaskStatus;
  createdAt: string;
  updatedAt: string;
  inputSummary: {
    videoName: string;
    dataFiles: string[];
  };
  resultSummary?: {
    jitterReductionRatio?: number;
    residualMse?: number;
    artifactCount: number;
  };
}

export interface VideoInfo {
  durationSec: number;
  fps: number;
  resolution: string;
  codec: string;
}

export interface DataOverview {
  samples: number;
  dimensions: string[];
  timeRangeSec: [number, number];
  hasPredictionSegment: boolean;
  notes: string;
}

export interface SeriesPoint {
  t: number;
  x?: number;
  y?: number;
  z?: number;
  residual?: number;
  translation?: number;
  rotation?: number;
}

export interface MetricsSummary {
  jitterReductionRatio: number;
  residualMse: number;
  maxDisplacementPx: number;
  avgRotationChangeDeg: number;
  cropStabilityScore: number;
}

export interface ResultArtifacts {
  stabilizedVideoUrl: string;
  compareVideoUrl?: string;
  keyframesActualCsvUrl: string;
  keyframesResidualCsvUrl: string;
  metricsJsonUrl: string;
  reportJsonUrl?: string;
  reportPdfUrl?: string;
}

export interface ExplainabilityOutput {
  compensationTrajectory: string;
  cropWindow: string;
  modelPlaceholder: string;
}

export interface TaskResult {
  metrics: MetricsSummary;
  artifacts: ResultArtifacts;
  explainability: ExplainabilityOutput;
  charts: {
    jitterTrajectory: SeriesPoint[];
    residualSeries: SeriesPoint[];
    keyframeSeries: SeriesPoint[];
    actualMotionTrajectory?: SeriesPoint[];
    idealMotionTrajectory?: SeriesPoint[];
    diffTrajectory?: SeriesPoint[];
  };
}

export interface TaskDetail {
  task: TaskSummary & {
    params: TaskParams;
    inputFiles: UploadedFileMeta[];
  };
  inputOverview: {
    video: VideoInfo;
    data: DataOverview;
  };
  result?: TaskResult;
}

export interface TaskStatusResponse {
  taskId: string;
  status: TaskStatus;
  progress: number;
  message: string;
  updatedAt: string;
}

export interface TaskResultResponse {
  taskId: string;
  status: TaskStatus;
  result?: TaskResult;
}

export interface TaskQuery {
  status?: TaskStatus;
  mode?: SceneMode;
}

export interface UploadFileRequest {
  file: File;
  purpose: FilePurpose;
}

export interface UploadFileResponse extends UploadedFileMeta {}

export interface EndpointContract {
  name: string;
  method: "GET" | "POST";
  path: string;
  description: string;
  requestType: string;
  responseType: string;
  requestExample?: unknown;
  responseExample?: unknown;
}

export interface FileFormatContract {
  name: string;
  scene: "static" | "dynamic" | "both";
  required: boolean;
  extensions: string[];
  columns: string[];
  description: string;
}

export interface ArtifactContract {
  name: string;
  required: boolean;
  description: string;
}

export interface DataFlowContract {
  scene: SceneMode;
  steps: string[];
}

export interface PythonScriptContract {
  command: string;
  requiredArgs: string[];
  optionalArgs: string[];
  outputs: string[];
}

export interface ContractsMetaResponse {
  version: string;
  generatedAt: string;
  endpoints: EndpointContract[];
  fileFormats: FileFormatContract[];
  artifacts: ArtifactContract[];
  flows: DataFlowContract[];
  pythonIntegration: PythonScriptContract;
}
