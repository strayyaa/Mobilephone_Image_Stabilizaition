import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export function NotFoundPage() {
  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <div className="w-full max-w-lg rounded-2xl border border-border bg-card/70 p-8 text-center">
        <h1 className="font-display text-3xl font-semibold">404</h1>
        <p className="mt-2 text-sm text-muted-foreground">页面不存在或路由已失效。</p>
        <Button asChild className="mt-5">
          <Link to="/">返回系统首页</Link>
        </Button>
      </div>
    </div>
  );
}
