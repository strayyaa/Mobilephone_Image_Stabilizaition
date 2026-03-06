import { PlaySquare } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface VideoComparePanelProps {
  beforeUrl?: string;
  afterUrl?: string;
}

function VideoBox({ label, src }: { label: string; src?: string }) {
  return (
    <div className="rounded-xl border border-border bg-background/50 p-3">
      <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      {src ? (
        <video className="h-56 w-full rounded-lg bg-black/60 object-cover md:h-72" controls src={src} />
      ) : (
        <div className="flex h-56 w-full items-center justify-center rounded-lg border border-dashed border-border bg-black/20 md:h-72">
          <div className="text-center">
            <PlaySquare className="mx-auto h-8 w-8 text-muted-foreground" />
            <p className="mt-2 text-sm text-muted-foreground">暂无视频</p>
          </div>
        </div>
      )}
    </div>
  );
}

export function VideoComparePanel({ beforeUrl, afterUrl }: VideoComparePanelProps) {
  return (
    <Card className="glass-card border-white/10">
      <CardHeader>
        <CardTitle className="text-base">稳像前后视频对比</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-2">
        <VideoBox label="处理前" src={beforeUrl} />
        <VideoBox label="处理后" src={afterUrl} />
      </CardContent>
    </Card>
  );
}
