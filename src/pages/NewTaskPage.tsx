import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { SectionHeader } from "@/components/common/SectionHeader";
import { UploadZone } from "@/components/forms/UploadZone";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useTaskStore } from "@/features/tasks/taskStore";
import type {
  CreateDynamicTaskRequest,
  CreateStaticTaskRequest,
  CropStrategy,
  OutputResolutionStrategy,
  UploadedFileMeta,
} from "@/types/contracts";

type StaticFormState = {
  name: string;
  description: string;
  sourceVideo?: UploadedFileMeta;
  jitterCsv?: UploadedFileMeta;
  predictionCsv?: UploadedFileMeta;
  compensationStrength: number;
  cropStrategy: CropStrategy;
  smoothingLevel: number;
  enableRotationCompensation: boolean;
  outputResolutionStrategy: OutputResolutionStrategy;
};

type DynamicFormState = {
  name: string;
  description: string;
  sourceVideo?: UploadedFileMeta;
  actualMotionCsv?: UploadedFileMeta;
  idealMotionCsv?: UploadedFileMeta;
  diffJitterCsv?: UploadedFileMeta;
  autoComputeDiff: boolean;
  compensationStrength: number;
  cropStrategy: CropStrategy;
  smoothingLevel: number;
  enableRotationCompensation: boolean;
};

const initialStaticForm: StaticFormState = {
  name: "",
  description: "",
  compensationStrength: 75,
  cropStrategy: "adaptive",
  smoothingLevel: 7,
  enableRotationCompensation: true,
  outputResolutionStrategy: "smart_crop",
};

const initialDynamicForm: DynamicFormState = {
  name: "",
  description: "",
  autoComputeDiff: true,
  compensationStrength: 72,
  cropStrategy: "adaptive",
  smoothingLevel: 7,
  enableRotationCompensation: true,
};

function ParamSlider({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (val: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span>{label}</span>
        <span className="font-mono text-muted-foreground">{value}</span>
      </div>
      <input
        type="range"
        className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-muted"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export function NewTaskPage() {
  const [searchParams] = useSearchParams();
  const defaultMode = searchParams.get("mode") === "dynamic" ? "dynamic" : "static";
  const [activeMode, setActiveMode] = useState<"static" | "dynamic">(defaultMode);
  const [staticForm, setStaticForm] = useState<StaticFormState>(initialStaticForm);
  const [dynamicForm, setDynamicForm] = useState<DynamicFormState>(initialDynamicForm);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string>();
  const navigate = useNavigate();
  const { createTaskAndRefresh } = useTaskStore();

  useEffect(() => {
    setActiveMode(defaultMode);
  }, [defaultMode]);

  const staticValid = useMemo(() => !!(staticForm.name && staticForm.sourceVideo && staticForm.jitterCsv), [staticForm]);
  const dynamicValid = useMemo(
    () => !!(dynamicForm.name && dynamicForm.sourceVideo && dynamicForm.actualMotionCsv && dynamicForm.idealMotionCsv),
    [dynamicForm]
  );

  async function submitStatic() {
    if (!staticValid || !staticForm.sourceVideo || !staticForm.jitterCsv) {
      setError("请补全静止场景必填字段。");
      return;
    }
    setError(undefined);
    setSubmitting(true);
    try {
      const payload: CreateStaticTaskRequest = {
        mode: "static",
        name: staticForm.name,
        description: staticForm.description || undefined,
        inputs: {
          sourceVideoFileId: staticForm.sourceVideo.fileId,
          jitterCsvFileId: staticForm.jitterCsv.fileId,
          predictionCsvFileId: staticForm.predictionCsv?.fileId,
        },
        params: {
          compensationStrength: staticForm.compensationStrength,
          cropStrategy: staticForm.cropStrategy,
          smoothingLevel: staticForm.smoothingLevel,
          enableRotationCompensation: staticForm.enableRotationCompensation,
          outputResolutionStrategy: staticForm.outputResolutionStrategy,
        },
      };
      const created = await createTaskAndRefresh(payload);
      navigate(`/tasks/${created.id}`);
    } finally {
      setSubmitting(false);
    }
  }

  async function submitDynamic() {
    if (!dynamicValid || !dynamicForm.sourceVideo || !dynamicForm.actualMotionCsv || !dynamicForm.idealMotionCsv) {
      setError("请补全运动场景必填字段。");
      return;
    }
    setError(undefined);
    setSubmitting(true);
    try {
      const payload: CreateDynamicTaskRequest = {
        mode: "dynamic",
        name: dynamicForm.name,
        description: dynamicForm.description || undefined,
        inputs: {
          sourceVideoFileId: dynamicForm.sourceVideo.fileId,
          actualMotionCsvFileId: dynamicForm.actualMotionCsv.fileId,
          idealMotionCsvFileId: dynamicForm.idealMotionCsv.fileId,
          diffJitterCsvFileId: dynamicForm.diffJitterCsv?.fileId,
        },
        params: {
          autoComputeDiff: dynamicForm.autoComputeDiff,
          compensationStrength: dynamicForm.compensationStrength,
          cropStrategy: dynamicForm.cropStrategy,
          smoothingLevel: dynamicForm.smoothingLevel,
          enableRotationCompensation: dynamicForm.enableRotationCompensation,
        },
      };
      const created = await createTaskAndRefresh(payload);
      navigate(`/tasks/${created.id}`);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-6">
      <SectionHeader
        title="新建处理任务"
        subtitle="支持静止场景与运动场景两种模式。当前页面仅对接 mock API，但字段已按后端契约完整保留。"
      />
      <Tabs value={activeMode} onValueChange={(v) => setActiveMode(v as "static" | "dynamic")}>
        <TabsList>
          <TabsTrigger value="static">静止场景</TabsTrigger>
          <TabsTrigger value="dynamic">运动场景</TabsTrigger>
        </TabsList>

        <TabsContent value="static">
          <Card className="glass-card border-white/10">
            <CardHeader>
              <CardTitle className="text-lg">静止场景任务配置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">任务名称 *</label>
                  <Input
                    placeholder="例如：静止场景-手机轻抖动测试"
                    value={staticForm.name}
                    onChange={(e) => setStaticForm((f) => ({ ...f, name: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">任务说明（可选）</label>
                  <Textarea
                    rows={2}
                    placeholder="简要描述实验目标"
                    value={staticForm.description}
                    onChange={(e) => setStaticForm((f) => ({ ...f, description: e.target.value }))}
                  />
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <UploadZone
                  title="原始视频文件"
                  hint="后端用于稳像处理与前后视频对比"
                  accept=".mp4,.mov,.avi,video/*"
                  purpose="source_video"
                  required
                  value={staticForm.sourceVideo}
                  onChange={(file) => setStaticForm((f) => ({ ...f, sourceVideo: file }))}
                />
                <UploadZone
                  title="抖动数据文件"
                  hint="建议字段: timestamp,x_true,y_true,z_true"
                  accept=".csv,text/csv"
                  purpose="jitter_csv"
                  required
                  value={staticForm.jitterCsv}
                  onChange={(file) => setStaticForm((f) => ({ ...f, jitterCsv: file }))}
                />
              </div>
              <UploadZone
                title="预测结果文件（可选）"
                hint="若离线模型已产出 prediction 段，可直接上传"
                accept=".csv,text/csv"
                purpose="prediction_csv"
                value={staticForm.predictionCsv}
                onChange={(file) => setStaticForm((f) => ({ ...f, predictionCsv: file }))}
              />

              <div className="grid gap-4 rounded-xl border border-border bg-card/40 p-4 md:grid-cols-2">
                <ParamSlider
                  label="补偿强度"
                  value={staticForm.compensationStrength}
                  min={0}
                  max={100}
                  onChange={(val) => setStaticForm((f) => ({ ...f, compensationStrength: val }))}
                />
                <ParamSlider
                  label="平滑级别"
                  value={staticForm.smoothingLevel}
                  min={1}
                  max={10}
                  onChange={(val) => setStaticForm((f) => ({ ...f, smoothingLevel: val }))}
                />
                <div className="space-y-2">
                  <label className="text-sm font-medium">裁切策略</label>
                  <select
                    className="h-10 w-full rounded-md border border-input bg-background/60 px-3 text-sm"
                    value={staticForm.cropStrategy}
                    onChange={(e) => setStaticForm((f) => ({ ...f, cropStrategy: e.target.value as CropStrategy }))}
                  >
                    <option value="adaptive">adaptive</option>
                    <option value="fixed">fixed</option>
                    <option value="none">none</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">输出分辨率策略</label>
                  <select
                    className="h-10 w-full rounded-md border border-input bg-background/60 px-3 text-sm"
                    value={staticForm.outputResolutionStrategy}
                    onChange={(e) =>
                      setStaticForm((f) => ({ ...f, outputResolutionStrategy: e.target.value as OutputResolutionStrategy }))
                    }
                  >
                    <option value="keep">keep</option>
                    <option value="smart_crop">smart_crop</option>
                    <option value="fixed_1080p">fixed_1080p</option>
                  </select>
                </div>
                <div className="md:col-span-2 flex items-center justify-between rounded-lg border border-border p-3">
                  <p className="text-sm">启用旋转补偿</p>
                  <Switch
                    checked={staticForm.enableRotationCompensation}
                    onCheckedChange={(checked) => setStaticForm((f) => ({ ...f, enableRotationCompensation: checked }))}
                  />
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                契约提示：后端将根据 `sourceVideoFileId + jitterCsvFileId (+predictionCsvFileId)` 拉取文件，并按 params 生成关键帧与稳像产物。
              </p>
              {error ? <p className="text-sm text-rose-300">{error}</p> : null}
              <div className="flex justify-end">
                <Button onClick={() => void submitStatic()} disabled={!staticValid || submitting}>
                  {submitting ? "提交中..." : "提交静止场景任务"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dynamic">
          <Card className="glass-card border-white/10">
            <CardHeader>
              <CardTitle className="text-lg">运动场景任务配置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">任务名称 *</label>
                  <Input
                    placeholder="例如：无人机飞行差分防抖任务"
                    value={dynamicForm.name}
                    onChange={(e) => setDynamicForm((f) => ({ ...f, name: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">任务说明（可选）</label>
                  <Textarea
                    rows={2}
                    placeholder="记录试验环境、路线或版本"
                    value={dynamicForm.description}
                    onChange={(e) => setDynamicForm((f) => ({ ...f, description: e.target.value }))}
                  />
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <UploadZone
                  title="原始视频文件"
                  hint="后端处理主体视频"
                  accept=".mp4,.mov,.avi,video/*"
                  purpose="source_video"
                  required
                  value={dynamicForm.sourceVideo}
                  onChange={(file) => setDynamicForm((f) => ({ ...f, sourceVideo: file }))}
                />
                <UploadZone
                  title="实际运动数据文件"
                  hint="建议字段: timestamp,x,y,z 或 roll/pitch/yaw"
                  accept=".csv,text/csv"
                  purpose="actual_motion_csv"
                  required
                  value={dynamicForm.actualMotionCsv}
                  onChange={(file) => setDynamicForm((f) => ({ ...f, actualMotionCsv: file }))}
                />
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <UploadZone
                  title="理想运动数据文件"
                  hint="用于与实际运动做差得到抖动"
                  accept=".csv,text/csv"
                  purpose="ideal_motion_csv"
                  required
                  value={dynamicForm.idealMotionCsv}
                  onChange={(file) => setDynamicForm((f) => ({ ...f, idealMotionCsv: file }))}
                />
                <UploadZone
                  title="差分抖动文件（可选）"
                  hint="若已离线计算差分，可跳过后端自动计算"
                  accept=".csv,text/csv"
                  purpose="diff_jitter_csv"
                  value={dynamicForm.diffJitterCsv}
                  onChange={(file) => setDynamicForm((f) => ({ ...f, diffJitterCsv: file }))}
                />
              </div>

              <div className="grid gap-4 rounded-xl border border-border bg-card/40 p-4 md:grid-cols-2">
                <div className="md:col-span-2 flex items-center justify-between rounded-lg border border-border p-3">
                  <div>
                    <p className="text-sm">由后端自动计算差分</p>
                    <p className="text-xs text-muted-foreground">开启后，后端将执行 actual - ideal</p>
                  </div>
                  <Switch checked={dynamicForm.autoComputeDiff} onCheckedChange={(v) => setDynamicForm((f) => ({ ...f, autoComputeDiff: v }))} />
                </div>
                <ParamSlider
                  label="补偿强度"
                  value={dynamicForm.compensationStrength}
                  min={0}
                  max={100}
                  onChange={(val) => setDynamicForm((f) => ({ ...f, compensationStrength: val }))}
                />
                <ParamSlider
                  label="平滑级别"
                  value={dynamicForm.smoothingLevel}
                  min={1}
                  max={10}
                  onChange={(val) => setDynamicForm((f) => ({ ...f, smoothingLevel: val }))}
                />
                <div className="space-y-2">
                  <label className="text-sm font-medium">裁切策略</label>
                  <select
                    className="h-10 w-full rounded-md border border-input bg-background/60 px-3 text-sm"
                    value={dynamicForm.cropStrategy}
                    onChange={(e) => setDynamicForm((f) => ({ ...f, cropStrategy: e.target.value as CropStrategy }))}
                  >
                    <option value="adaptive">adaptive</option>
                    <option value="fixed">fixed</option>
                    <option value="none">none</option>
                  </select>
                </div>
                <div className="flex items-center justify-between rounded-lg border border-border p-3">
                  <p className="text-sm">启用旋转补偿</p>
                  <Switch
                    checked={dynamicForm.enableRotationCompensation}
                    onCheckedChange={(checked) => setDynamicForm((f) => ({ ...f, enableRotationCompensation: checked }))}
                  />
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                契约提示：当 `autoComputeDiff=true` 且未传 diffJitterCsvFileId 时，后端需先执行差分阶段，再进入稳像处理。
              </p>
              {error ? <p className="text-sm text-rose-300">{error}</p> : null}
              <div className="flex justify-end">
                <Button onClick={() => void submitDynamic()} disabled={!dynamicValid || submitting}>
                  {submitting ? "提交中..." : "提交运动场景任务"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
