import { useEffect, useState } from "react";
import { ApiContractTable } from "@/components/dev/ApiContractTable";
import { EmptyState } from "@/components/common/EmptyState";
import { SectionHeader } from "@/components/common/SectionHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { getContractsMeta } from "@/mock/api";
import type { ContractsMetaResponse } from "@/types/contracts";

export function DeveloperContractsPage() {
  const [meta, setMeta] = useState<ContractsMetaResponse>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  useEffect(() => {
    setLoading(true);
    void getContractsMeta()
      .then((res) => setMeta(res))
      .catch((err) => setError(err instanceof Error ? err.message : "加载失败"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="rounded-xl border border-border bg-card/40 p-6 text-sm text-muted-foreground">正在加载接口契约...</div>;
  }

  if (error || !meta) {
    return <EmptyState title="无法加载契约信息" description={error ?? "请稍后再试。"} />;
  }

  return (
    <div className="space-y-6">
      <SectionHeader
        title="数据接口说明 / 开发者契约"
        subtitle="该页面面向后端开发者，明确请求字段、文件格式与结果产物。当前由 /api/meta/contracts 提供数据。"
        actions={<Badge variant="outline">Contract {meta.version}</Badge>}
      />

      <ApiContractTable endpoints={meta.endpoints} />

      <Card className="glass-card border-white/10">
        <CardHeader>
          <CardTitle className="text-base">请求/响应示例</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {meta.endpoints.map((endpoint) => (
            <div key={endpoint.path} className="rounded-lg border border-border bg-background/40 p-3">
              <p className="mb-2 text-sm font-semibold">
                {endpoint.method} {endpoint.path}
              </p>
              <p className="mb-2 text-xs text-muted-foreground">{endpoint.description}</p>
              {endpoint.requestExample ? (
                <pre className="scrollbar-thin overflow-auto rounded-md border border-border bg-black/20 p-3 text-xs">
                  {JSON.stringify(endpoint.requestExample, null, 2)}
                </pre>
              ) : null}
              {endpoint.responseExample ? (
                <pre className="scrollbar-thin mt-2 overflow-auto rounded-md border border-border bg-black/20 p-3 text-xs">
                  {JSON.stringify(endpoint.responseExample, null, 2)}
                </pre>
              ) : null}
            </div>
          ))}
        </CardContent>
      </Card>

      <section className="grid gap-4 xl:grid-cols-2">
        <Card className="glass-card border-white/10">
          <CardHeader>
            <CardTitle className="text-base">输入文件格式约定</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {meta.fileFormats.map((file) => (
              <div key={file.name} className="rounded-lg border border-border bg-background/40 p-3">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <p className="text-sm font-semibold">{file.name}</p>
                  <Badge variant="outline">{file.scene}</Badge>
                  <Badge variant={file.required ? "secondary" : "outline"}>{file.required ? "required" : "optional"}</Badge>
                </div>
                <p className="text-xs text-muted-foreground">extensions: {file.extensions.join(", ")}</p>
                <p className="mt-1 text-xs text-muted-foreground">columns: {file.columns.join(", ") || "-"}</p>
                <p className="mt-2 text-xs">{file.description}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="glass-card border-white/10">
          <CardHeader>
            <CardTitle className="text-base">结果文件与数据流</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="mb-2 text-sm font-semibold">结果文件</p>
              <div className="space-y-2">
                {meta.artifacts.map((artifact) => (
                  <div key={artifact.name} className="rounded-lg border border-border bg-background/40 p-3 text-sm">
                    <p className="font-mono text-xs text-sky-300">{artifact.name}</p>
                    <p className="mt-1 text-xs text-muted-foreground">{artifact.description}</p>
                  </div>
                ))}
              </div>
            </div>
            <Separator />
            <div>
              <p className="mb-2 text-sm font-semibold">场景数据流</p>
              <div className="space-y-2">
                {meta.flows.map((flow) => (
                  <div key={flow.scene} className="rounded-lg border border-border bg-background/40 p-3">
                    <div className="mb-2 flex items-center gap-2">
                      <Badge variant="outline">{flow.scene}</Badge>
                    </div>
                    <ol className="list-decimal space-y-1 pl-5 text-xs text-muted-foreground">
                      {flow.steps.map((step) => (
                        <li key={step}>{step}</li>
                      ))}
                    </ol>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <Card className="glass-card border-white/10">
        <CardHeader>
          <CardTitle className="text-base">Python 脚本接入说明（后端预留）</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p>command: <code>{meta.pythonIntegration.command}</code></p>
          <p>required args: <code>{meta.pythonIntegration.requiredArgs.join(", ")}</code></p>
          <p>optional args: <code>{meta.pythonIntegration.optionalArgs.join(", ")}</code></p>
          <p>outputs: <code>{meta.pythonIntegration.outputs.join(", ")}</code></p>
          <p className="text-xs text-muted-foreground">
            后端建议把脚本产物映射到 `/api/tasks/:id/result` 的 `artifacts` 字段，并通过 `/download/:artifactName` 对外提供下载。
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
