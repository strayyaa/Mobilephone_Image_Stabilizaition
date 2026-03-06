import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Copy, Download } from "lucide-react";
import { useParams } from "react-router-dom";
import { MotionChart } from "@/components/charts/MotionChart";
import { EmptyState } from "@/components/common/EmptyState";
import { MetricCard } from "@/components/common/MetricCard";
import { SectionHeader } from "@/components/common/SectionHeader";
import { StatusBadge } from "@/components/common/StatusBadge";
import { ResultArtifactsPanel } from "@/components/detail/ResultArtifactsPanel";
import { VideoComparePanel } from "@/components/detail/VideoComparePanel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { downloadTaskArtifact, getTaskStatus } from "@/mock/api";
import { useTaskStore } from "@/features/tasks/taskStore";
import { useInterval } from "@/hooks/useInterval";
import { formatDate, formatPercent } from "@/lib/format";

const modeLabelMap = {
  static: "静止场景",
  dynamic: "运动场景",
} as const;

export function TaskDetailPage() {
  const { taskId } = useParams();
  const { selectedTask, fetchTaskDetail, loading, error } = useTaskStore();
  const [statusText, setStatusText] = useState<string>();
  const [polling, setPolling] = useState(false);

  useEffect(() => {
    if (!taskId) return;
    void fetchTaskDetail(taskId);
  }, [fetchTaskDetail, taskId]);

  useEffect(() => {
    setPolling(!!(taskId && selectedTask && selectedTask.task.status === "processing"));
    return () => setPolling(false);
  }, [selectedTask, taskId]);

  useInterval(
    () => {
      if (!taskId || !selectedTask || selectedTask.task.status !== "processing") return;
      void getTaskStatus(taskId).then((status) => setStatusText(`${status.message} (${status.progress}%)`));
    },
    taskId && selectedTask?.task.status === "processing" ? 2500 : null
  );

  async function handleDownload(artifactName: string) {
    if (!taskId) return;
    const url = await downloadTaskArtifact(taskId, artifactName);
    window.open(url, "_blank", "noopener,noreferrer");
  }

  const paramsSummary = useMemo(() => {
    if (!selectedTask) return [];
    const params = selectedTask.task.params;
    return Object.entries(params).map(([key, value]) => ({ key, value: String(value) }));
  }, [selectedTask]);

  if (!taskId) {
    return <EmptyState title="任务不存在" description="URL 中缺少 taskId。" />;
  }

  if (loading && !selectedTask) {
    return <div className="rounded-xl border border-border bg-card/40 p-6 text-sm text-muted-foreground">正在加载任务详情...</div>;
  }

  if (error && !selectedTask) {
    return (
      <div className="rounded-xl border border-rose-400/40 bg-rose-500/10 p-6 text-sm text-rose-200">
        加载失败：{error}
      </div>
    );
  }

  if (!selectedTask) {
    return <EmptyState title="未找到任务" description="该任务可能已被删除或尚未创建。" />;
  }

  const { task, inputOverview, result } = selectedTask;

  return (
    <div className="space-y-6">
      <SectionHeader
        title={task.name}
        subtitle={`任务ID: ${task.id}`}
        actions={
          <div className="flex gap-2">
            <Badge variant="outline">{modeLabelMap[task.mode]}</Badge>
            <StatusBadge status={task.status} />
          </div>
        }
      />

      <section className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <Card className="glass-card border-white/10">
          <CardHeader>
            <CardTitle className="text-base">A. 任务基本信息</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3 text-sm md:grid-cols-2">
            <p>创建时间: {formatDate(task.createdAt)}</p>
            <p>更新时间: {formatDate(task.updatedAt)}</p>
            <p>场景类型: {modeLabelMap[task.mode]}</p>
            <p>状态: {task.status}</p>
            <div className="md:col-span-2">
              <p className="mb-1 text-muted-foreground">输入文件</p>
              <div className="flex flex-wrap gap-2">
                {task.inputFiles.length ? (
                  task.inputFiles.map((file) => (
                    <Badge key={file.fileId} variant="outline">
                      {file.fileName}
                    </Badge>
                  ))
                ) : (
                  <span className="text-muted-foreground">暂无</span>
                )}
              </div>
            </div>
            <div className="md:col-span-2">
              <p className="mb-2 text-muted-foreground">参数摘要</p>
              <div className="grid gap-2 md:grid-cols-2">
                {paramsSummary.map((item) => (
                  <div key={item.key} className="rounded-md border border-border bg-background/30 px-3 py-2 font-mono text-xs">
                    {item.key}: {item.value}
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-white/10">
          <CardHeader>
            <CardTitle className="text-base">B. 输入数据概览</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p>视频时长: {inputOverview.video.durationSec}s</p>
            <p>视频帧率: {inputOverview.video.fps} FPS</p>
            <p>分辨率: {inputOverview.video.resolution}</p>
            <p>编码: {inputOverview.video.codec}</p>
            <p>样本点数: {inputOverview.data.samples}</p>
            <p>数据维度: {inputOverview.data.dimensions.join(", ") || "-"}</p>
            <p>
              时间范围: {inputOverview.data.timeRangeSec[0]} - {inputOverview.data.timeRangeSec[1]} s
            </p>
            <p>prediction 段: {inputOverview.data.hasPredictionSegment ? "存在" : "不存在"}</p>
            <p className="rounded-lg border border-border bg-background/40 p-2 text-xs text-muted-foreground">{inputOverview.data.notes}</p>
          </CardContent>
        </Card>
      </section>

      {!result ? (
        <Card className="glass-card border-white/10">
          <CardContent className="flex flex-col items-center justify-center gap-2 py-12 text-center">
            <AlertTriangle className="h-8 w-8 text-amber-300" />
            <p className="text-lg font-semibold">任务结果尚未就绪</p>
            <p className="text-sm text-muted-foreground">
              当前状态：{task.status}
              {statusText ? ` · ${statusText}` : ""}
            </p>
            {polling ? <p className="text-xs text-muted-foreground">已启动轮询 `/api/tasks/:id/status`。</p> : null}
          </CardContent>
        </Card>
      ) : (
        <>
          <section className="space-y-4">
            <SectionHeader title="C. 核心结果展示" subtitle="视频对比、轨迹图、关键帧与指标结果。" />
            <VideoComparePanel beforeUrl={result.artifacts.compareVideoUrl} afterUrl={result.artifacts.stabilizedVideoUrl} />
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              <MetricCard label="抖动幅度下降比例" value={formatPercent(result.metrics.jitterReductionRatio)} />
              <MetricCard label="残差均方值" value={result.metrics.residualMse.toFixed(4)} />
              <MetricCard label="最大位移" value={`${result.metrics.maxDisplacementPx.toFixed(2)} px`} />
              <MetricCard label="平均旋转变化" value={`${result.metrics.avgRotationChangeDeg.toFixed(2)} deg`} />
              <MetricCard label="裁切稳定性评分" value={result.metrics.cropStabilityScore.toFixed(2)} />
            </div>
            <div className="grid gap-4 xl:grid-cols-2">
              <MotionChart
                title="抖动轨迹图（x / y / z）"
                data={result.charts.jitterTrajectory}
                lines={[
                  { key: "x", name: "X", color: "#38bdf8" },
                  { key: "y", name: "Y", color: "#22d3ee" },
                  { key: "z", name: "Z", color: "#a78bfa" },
                ]}
              />
              <MotionChart title="补偿后残差图" data={result.charts.residualSeries} lines={[{ key: "residual", name: "Residual", color: "#f97316" }]} />
            </div>
            <MotionChart
              title="平移与旋转关键帧图"
              data={result.charts.keyframeSeries}
              lines={[
                { key: "translation", name: "Translation", color: "#2dd4bf" },
                { key: "rotation", name: "Rotation", color: "#f43f5e" },
              ]}
            />
          </section>

          <section className="grid gap-4 xl:grid-cols-[1.1fr_1fr]">
            <Card className="glass-card border-white/10">
              <CardHeader>
                <CardTitle className="text-base">可解释输出区域</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <div className="rounded-lg border border-border bg-background/40 p-3">
                  <p className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">补偿轨迹说明</p>
                  <p>{result.explainability.compensationTrajectory}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/40 p-3">
                  <p className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">裁切窗口说明</p>
                  <p>{result.explainability.cropWindow}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/40 p-3">
                  <p className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">模型/预测结果占位</p>
                  <p>{result.explainability.modelPlaceholder}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button asChild variant="outline" size="sm">
                    <a href={result.artifacts.keyframesActualCsvUrl} target="_blank" rel="noreferrer">
                      <Download className="mr-1 h-4 w-4" />
                      关键帧 CSV
                    </a>
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      void navigator.clipboard.writeText(task.id);
                    }}
                  >
                    <Copy className="mr-1 h-4 w-4" />
                    复制任务ID
                  </Button>
                </div>
              </CardContent>
            </Card>
            <ResultArtifactsPanel taskId={task.id} artifacts={result.artifacts} onDownload={handleDownload} />
          </section>
        </>
      )}

      {task.mode === "dynamic" && result ? (
        <section className="space-y-4">
          <SectionHeader title="D. 运动场景扩展展示" subtitle="实际轨迹、理想轨迹与差分抖动的可解释对比。" />
          <div className="grid gap-4 xl:grid-cols-3">
            {result.charts.actualMotionTrajectory ? (
              <MotionChart
                title="实际运动轨迹"
                data={result.charts.actualMotionTrajectory}
                lines={[
                  { key: "x", name: "X", color: "#38bdf8" },
                  { key: "y", name: "Y", color: "#22d3ee" },
                  { key: "z", name: "Z", color: "#818cf8" },
                ]}
              />
            ) : null}
            {result.charts.idealMotionTrajectory ? (
              <MotionChart
                title="理想运动轨迹"
                data={result.charts.idealMotionTrajectory}
                lines={[
                  { key: "x", name: "X", color: "#4ade80" },
                  { key: "y", name: "Y", color: "#34d399" },
                  { key: "z", name: "Z", color: "#10b981" },
                ]}
              />
            ) : null}
            {result.charts.diffTrajectory ? (
              <MotionChart
                title="差分抖动轨迹"
                data={result.charts.diffTrajectory}
                lines={[
                  { key: "x", name: "dX", color: "#f59e0b" },
                  { key: "y", name: "dY", color: "#fb7185" },
                  { key: "z", name: "dZ", color: "#f97316" },
                ]}
              />
            ) : null}
          </div>
          <Card className="glass-card border-white/10">
            <CardContent className="pt-6 text-sm text-muted-foreground">
              差分抖动说明：后端对齐 `actual_motion` 与 `ideal_motion` 后进行逐时刻相减，得到抖动项输入补偿模块。
            </CardContent>
          </Card>
        </section>
      ) : null}
    </div>
  );
}
