import { ArrowRight, CalendarClock, FolderArchive } from "lucide-react";
import { Link } from "react-router-dom";
import { StatusBadge } from "@/components/common/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatDate } from "@/lib/format";
import type { TaskSummary } from "@/types/contracts";

interface TaskCardProps {
  task: TaskSummary;
}

const modeLabelMap = {
  static: "静止场景",
  dynamic: "运动场景",
} as const;

export function TaskCard({ task }: TaskCardProps) {
  return (
    <Card className="glass-card border-white/10">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <Badge variant="outline">{modeLabelMap[task.mode]}</Badge>
          <StatusBadge status={task.status} />
        </div>
        <CardTitle className="pt-2 text-lg">{task.name}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <CalendarClock className="h-4 w-4" />
            <span>{formatDate(task.createdAt)}</span>
          </div>
          <div className="flex items-start gap-2">
            <FolderArchive className="mt-0.5 h-4 w-4" />
            <span className="line-clamp-2">{task.inputSummary.dataFiles.join(" / ") || "暂无数据文件"}</span>
          </div>
        </div>
        <div className="flex justify-end">
          <Button asChild size="sm" variant="secondary">
            <Link to={`/tasks/${task.id}`}>
              查看详情
              <ArrowRight className="ml-1 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
