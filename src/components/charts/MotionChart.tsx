import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { SeriesPoint } from "@/types/contracts";

interface MotionChartProps {
  title: string;
  data: SeriesPoint[];
  lines: {
    key: keyof SeriesPoint;
    name: string;
    color: string;
  }[];
  height?: number;
}

export function MotionChart({ title, data, lines, height = 280 }: MotionChartProps) {
  return (
    <Card className="glass-card border-white/10">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="h-[300px] p-4 pt-0">
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
            <XAxis dataKey="t" stroke="#94a3b8" fontSize={12} />
            <YAxis stroke="#94a3b8" fontSize={12} />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(17, 24, 39, 0.92)",
                border: "1px solid rgba(148, 163, 184, 0.25)",
                borderRadius: "10px",
              }}
            />
            <Legend />
            {lines.map((line) => (
              <Line
                key={String(line.key)}
                type="monotone"
                dataKey={line.key}
                name={line.name}
                stroke={line.color}
                dot={false}
                strokeWidth={2}
                isAnimationActive
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
