import { motion } from "framer-motion";
import { ArrowRight, BrainCircuit, ChartBarBig, Files, Gauge, Video } from "lucide-react";
import { useEffect } from "react";
import { Link } from "react-router-dom";
import { MetricCard } from "@/components/common/MetricCard";
import { SectionHeader } from "@/components/common/SectionHeader";
import { TaskCard } from "@/components/tasks/TaskCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useTaskStore } from "@/features/tasks/taskStore";
import { formatPercent } from "@/lib/format";

const modeEntries = [
  {
    mode: "static",
    title: "静止场景防抖",
    desc: "面向手机静止拍摄、无人机悬停等场景，直接上传抖动轨迹进行补偿展示。",
    requires: ["原始视频", "抖动CSV", "可选预测CSV"],
  },
  {
    mode: "dynamic",
    title: "运动场景防抖",
    desc: "上传实际运动与理想运动数据，由后端对齐差分后进入同类稳像流程。",
    requires: ["原始视频", "实际运动CSV", "理想运动CSV", "可选差分CSV"],
  },
] as const;

const flowSteps = ["输入上传", "任务创建", "后端处理", "结果可视化", "导出产物"];

export function OverviewPage() {
  const { tasks, fetchTasks } = useTaskStore();

  useEffect(() => {
    void fetchTasks();
  }, [fetchTasks]);

  const completed = tasks.filter((item) => item.status === "completed");
  const avgReduction = completed.length
    ? completed.reduce((sum, t) => sum + (t.resultSummary?.jitterReductionRatio ?? 0), 0) / completed.length
    : 0;

  return (
    <div className="space-y-8">
      <section className="rounded-2xl border border-white/10 bg-hero-grid p-7 shadow-glow">
        <Badge variant="secondary" className="mb-4">
          AI Vision Motion Stabilization
        </Badge>
        <h2 className="max-w-3xl font-display text-3xl font-semibold leading-tight md:text-4xl">
          视频防抖展示与处理平台
        </h2>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-muted-foreground md:text-base">
          聚焦“基于过去抖动信息预测未来抖动并反向补偿”的工程化展示。当前版本仅实现前端流程与接口契约，支持本地 mock 演示与后续 Python
          后端无缝对接。
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Button asChild>
            <Link to="/tasks/new">
              开始新建任务
              <ArrowRight className="ml-1 h-4 w-4" />
            </Link>
          </Button>
          <Button asChild variant="outline">
            <Link to="/developer/contracts">查看接口契约</Link>
          </Button>
        </div>
      </section>

      <section>
        <SectionHeader title="核心指标" subtitle="以下数据为 mock 示例，用于答辩展示页面效果。" />
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="已创建任务数" value={`${tasks.length}`} hint="含静止场景与运动场景" />
          <MetricCard label="完成任务平均稳像提升" value={formatPercent(avgReduction)} hint="抖动幅度下降比例" />
          <MetricCard label="当前处理中任务" value={`${tasks.filter((t) => t.status === "processing").length}`} hint="支持状态轮询" />
          <MetricCard label="可导出产物类型" value="6+" hint="视频、关键帧、报告、指标" />
        </div>
      </section>

      <section className="grid gap-5 xl:grid-cols-2">
        {modeEntries.map((entry, index) => (
          <motion.div
            key={entry.mode}
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: index * 0.08 }}
          >
            <Card className="glass-card h-full border-white/10">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  {entry.mode === "static" ? <Video className="h-5 w-5 text-primary" /> : <BrainCircuit className="h-5 w-5 text-accent" />}
                  {entry.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">{entry.desc}</p>
                <div className="flex flex-wrap gap-2">
                  {entry.requires.map((item) => (
                    <Badge key={item} variant="outline">
                      {item}
                    </Badge>
                  ))}
                </div>
                <Button asChild variant="secondary">
                  <Link to={`/tasks/new?mode=${entry.mode}`}>进入该模式</Link>
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.2fr_1fr]">
        <Card className="glass-card border-white/10">
          <CardHeader>
            <CardTitle className="text-base">处理流程示意</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-5">
              {flowSteps.map((step, idx) => (
                <div key={step} className="flex items-center gap-3">
                  <div className="flex h-11 w-11 items-center justify-center rounded-full border border-border bg-card">
                    {idx === 0 ? <Files className="h-4 w-4 text-primary" /> : idx === 1 ? <Gauge className="h-4 w-4 text-primary" /> : <ChartBarBig className="h-4 w-4 text-primary" />}
                  </div>
                  <div className="text-sm">{step}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card className="glass-card border-white/10">
          <CardHeader>
            <CardTitle className="text-base">系统亮点</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>1. 同时支持静止场景与运动场景，模式清晰切换。</p>
            <p>2. 上传、处理、可视化、导出全链路完整。</p>
            <p>3. 轨迹、残差、关键帧图表便于解释模型效果。</p>
            <p>4. 提供完整后端接口契约与文件格式规范。</p>
          </CardContent>
        </Card>
      </section>

      <section>
        <SectionHeader title="近期任务" subtitle="点击任务可进入详情页查看图表、视频对比与导出。" />
        <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
          {tasks.slice(0, 3).map((task) => (
            <TaskCard key={task.id} task={task} />
          ))}
        </div>
      </section>
    </div>
  );
}
