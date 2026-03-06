import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { EndpointContract } from "@/types/contracts";

interface ApiContractTableProps {
  endpoints: EndpointContract[];
}

export function ApiContractTable({ endpoints }: ApiContractTableProps) {
  return (
    <Card className="glass-card border-white/10">
      <CardHeader>
        <CardTitle className="text-base">接口总览</CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full min-w-[880px] text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-3 py-3">接口</th>
              <th className="px-3 py-3">Method</th>
              <th className="px-3 py-3">Path</th>
              <th className="px-3 py-3">请求类型</th>
              <th className="px-3 py-3">返回类型</th>
            </tr>
          </thead>
          <tbody>
            {endpoints.map((endpoint) => (
              <tr key={endpoint.path} className="border-t border-border/70">
                <td className="px-3 py-3 font-medium">{endpoint.name}</td>
                <td className="px-3 py-3">{endpoint.method}</td>
                <td className="px-3 py-3 font-mono text-xs text-sky-300">{endpoint.path}</td>
                <td className="px-3 py-3">{endpoint.requestType}</td>
                <td className="px-3 py-3">{endpoint.responseType}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
