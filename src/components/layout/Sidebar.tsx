import { Film, FileCode2, Home, PlusCircle, Rows3 } from "lucide-react";
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";

const menus = [
  { to: "/", label: "总览", icon: Home, end: true },
  { to: "/tasks/new", label: "新建任务", icon: PlusCircle },
  { to: "/tasks", label: "任务列表", icon: Rows3 },
  { to: "/developer/contracts", label: "开发者契约", icon: FileCode2 },
];

export function Sidebar() {
  return (
    <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-white/10 bg-black/20 p-5 backdrop-blur-xl lg:flex">
      <div className="mb-8 flex items-center gap-3">
        <div className="rounded-lg bg-primary/20 p-2 text-primary">
          <Film className="h-5 w-5" />
        </div>
        <div>
          <p className="font-display text-sm text-muted-foreground">Vision Motion Lab</p>
          <p className="font-display text-lg font-semibold">智能稳像平台</p>
        </div>
      </div>
      <nav className="space-y-2">
        {menus.map((menu) => (
          <NavLink
            key={menu.to}
            to={menu.to}
            end={menu.end}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all",
                isActive ? "bg-primary/20 text-primary shadow-glow" : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
              )
            }
          >
            <menu.icon className="h-4 w-4" />
            <span>{menu.label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto rounded-xl border border-border bg-card/50 p-4">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">系统定位</p>
        <p className="mt-2 text-sm leading-6 text-slate-200">
          展示“输入 → 处理 → 结果 → 导出”的防抖闭环，为后端算法服务提供可直接对接的接口契约。
        </p>
      </div>
    </aside>
  );
}
