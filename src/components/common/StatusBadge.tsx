import { Badge } from "@/components/ui/badge";
import type { TaskStatus } from "@/types/contracts";

interface StatusBadgeProps {
  status: TaskStatus;
}

const statusLabelMap: Record<TaskStatus, string> = {
  pending: "待处理",
  processing: "处理中",
  completed: "已完成",
  failed: "失败",
};

const statusVariantMap: Record<TaskStatus, "secondary" | "warning" | "success" | "danger"> = {
  pending: "secondary",
  processing: "warning",
  completed: "success",
  failed: "danger",
};

export function StatusBadge({ status }: StatusBadgeProps) {
  return <Badge variant={statusVariantMap[status]}>{statusLabelMap[status]}</Badge>;
}
