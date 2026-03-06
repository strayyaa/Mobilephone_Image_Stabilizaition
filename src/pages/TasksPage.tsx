import { useEffect, useState } from "react";
import { EmptyState } from "@/components/common/EmptyState";
import { SectionHeader } from "@/components/common/SectionHeader";
import { TaskCard } from "@/components/tasks/TaskCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useTaskStore } from "@/features/tasks/taskStore";
import type { SceneMode, TaskQuery, TaskStatus } from "@/types/contracts";

const statusTabs: { key: TaskStatus | "all"; label: string }[] = [
  { key: "all", label: "全部" },
  { key: "pending", label: "待处理" },
  { key: "processing", label: "处理中" },
  { key: "completed", label: "已完成" },
  { key: "failed", label: "失败" },
];

const modeTabs: { key: SceneMode | "all"; label: string }[] = [
  { key: "all", label: "全部模式" },
  { key: "static", label: "静止场景" },
  { key: "dynamic", label: "运动场景" },
];

export function TasksPage() {
  const { tasks, loading, fetchTasks } = useTaskStore();
  const [statusFilter, setStatusFilter] = useState<TaskStatus | "all">("all");
  const [modeFilter, setModeFilter] = useState<SceneMode | "all">("all");

  useEffect(() => {
    const query: TaskQuery = {
      status: statusFilter === "all" ? undefined : statusFilter,
      mode: modeFilter === "all" ? undefined : modeFilter,
    };
    void fetchTasks(query);
  }, [fetchTasks, modeFilter, statusFilter]);

  return (
    <div className="space-y-6">
      <SectionHeader title="任务列表" subtitle="支持按状态与场景类型过滤，点击卡片进入任务详情页。" />
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-border bg-card/40 p-4">
        <div className="flex items-center gap-2">
          <Badge variant="outline">状态筛选</Badge>
          {statusTabs.map((tab) => (
            <Button key={tab.key} variant={statusFilter === tab.key ? "secondary" : "ghost"} size="sm" onClick={() => setStatusFilter(tab.key)}>
              {tab.label}
            </Button>
          ))}
        </div>
        <div className="h-6 w-px bg-border" />
        <div className="flex items-center gap-2">
          <Badge variant="outline">模式筛选</Badge>
          {modeTabs.map((tab) => (
            <Button key={tab.key} variant={modeFilter === tab.key ? "secondary" : "ghost"} size="sm" onClick={() => setModeFilter(tab.key)}>
              {tab.label}
            </Button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="rounded-xl border border-border bg-card/40 p-6 text-sm text-muted-foreground">正在加载任务...</div>
      ) : tasks.length === 0 ? (
        <EmptyState title="没有符合条件的任务" description="你可以先创建一个静止场景或运动场景任务。" />
      ) : (
        <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
          {tasks.map((task) => (
            <TaskCard key={task.id} task={task} />
          ))}
        </div>
      )}
    </div>
  );
}
