import { CalendarDays, Sparkles } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";

const mapPathTitle: Record<string, string> = {
  "/": "系统总览",
  "/tasks/new": "新建任务",
  "/tasks": "任务列表",
  "/developer/contracts": "接口与契约",
};

export function Topbar() {
  const { pathname } = useLocation();
  const title = mapPathTitle[pathname] ?? "任务详情";
  return (
    <header className="sticky top-0 z-20 border-b border-white/10 bg-background/75 px-4 py-4 backdrop-blur-xl md:px-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="font-display text-xl font-semibold">{title}</h1>
          <p className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
            <CalendarDays className="h-3.5 w-3.5" />
            {new Date().toLocaleString("zh-CN")}
          </p>
        </div>
        <Button asChild variant="secondary" size="sm">
          <Link to="/tasks/new">
            <Sparkles className="mr-1 h-4 w-4" />
            创建新任务
          </Link>
        </Button>
      </div>
    </header>
  );
}
