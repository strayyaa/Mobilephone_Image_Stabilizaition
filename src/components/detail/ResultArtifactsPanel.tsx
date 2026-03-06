import { Download, FileJson2, FileSpreadsheet, Film } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ResultArtifacts } from "@/types/contracts";

interface ResultArtifactsPanelProps {
  taskId: string;
  artifacts: ResultArtifacts;
  onDownload: (artifactName: string) => Promise<void>;
}

export function ResultArtifactsPanel({ taskId, artifacts, onDownload }: ResultArtifactsPanelProps) {
  const items = [
    { key: "stabilizedVideo", label: "下载稳像后视频", icon: Film, url: artifacts.stabilizedVideoUrl },
    { key: "keyframesActual", label: "下载关键帧 Actual CSV", icon: FileSpreadsheet, url: artifacts.keyframesActualCsvUrl },
    { key: "keyframesResidual", label: "下载关键帧 Residual CSV", icon: FileSpreadsheet, url: artifacts.keyframesResidualCsvUrl },
    { key: "metrics", label: "下载 metrics.json", icon: FileJson2, url: artifacts.metricsJsonUrl },
    { key: "report", label: "下载任务报告", icon: Download, url: artifacts.reportPdfUrl ?? artifacts.reportJsonUrl },
  ].filter((item) => !!item.url);

  return (
    <Card className="glass-card border-white/10">
      <CardHeader>
        <CardTitle className="text-base">结果导出</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {items.map((item) => (
          <div key={item.key} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border p-3">
            <div className="flex items-center gap-2">
              <item.icon className="h-4 w-4 text-primary" />
              <span className="text-sm">{item.label}</span>
            </div>
            <div className="flex gap-2">
              <Button asChild variant="outline" size="sm">
                <a href={item.url} target="_blank" rel="noreferrer">
                  直链
                </a>
              </Button>
              <Button size="sm" onClick={() => void onDownload(item.key)}>
                下载
              </Button>
            </div>
          </div>
        ))}
        <p className="text-xs text-muted-foreground">taskId: {taskId}。后端可在下载接口做鉴权、签名与临时链接控制。</p>
      </CardContent>
    </Card>
  );
}
