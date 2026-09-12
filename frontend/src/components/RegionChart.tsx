import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { bestSoFar, REGIONS } from "../data/results";

interface RegionChartProps {
  taskIdx: number;
  matrix: (number | null)[][];
  color: string;
  pale: string;
}

interface Row {
  name: string;
  seen: boolean;
  acc: number | null;
  best: number | null;
  drop: number | null;
  retained: number | null;
  lost: number | null;
  never: number | null;
}

function buildRows(matrix: (number | null)[][], taskIdx: number): Row[] {
  return REGIONS.map((region, i) => {
    if (taskIdx < i) {
      return {
        name: `${region.short} · ${region.label}`,
        seen: false,
        acc: null,
        best: null,
        drop: null,
        retained: null,
        lost: null,
        never: 100,
      };
    }
    const acc = matrix[taskIdx]?.[i] ?? null;
    const best = bestSoFar(matrix, i, taskIdx);
    const drop = acc !== null && best !== null ? Math.max(0, best - acc) : null;
    return {
      name: `${region.short} · ${region.label.toLowerCase()}`,
      seen: true,
      acc,
      best,
      drop,
      retained: acc,
      lost: drop,
      never: null,
    };
  });
}

interface TooltipRowProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload: Row }>;
}

function Tip({ active, payload }: TooltipRowProps) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  if (!row.seen) {
    return <div className="chart-tip">learned in a later task — not evaluated yet.</div>;
  }
  const dropping = row.drop !== null && row.drop > 0.75;
  return (
    <div className="chart-tip">
      <div className="tip-title">{row.name}</div>
      <div>acc&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{Math.round(row.acc!)}%</div>
      <div>best so far {Math.round(row.best!)}%</div>
      <div className={dropping ? "bad" : "ok"}>
        {dropping
          ? `forgetting −${Math.round(row.drop!)}% vs best`
          : "no forgetting on this region"}
      </div>
    </div>
  );
}

const mono = "ui-monospace, SF Mono, Menlo, Consolas, monospace";

export default function RegionChart({ taskIdx, matrix, color, pale }: RegionChartProps) {
  const data = buildRows(matrix, taskIdx);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart
        data={data}
        margin={{ top: 18, right: 14, left: 2, bottom: 4 }}
        barCategoryGap="30%"
      >
        <CartesianGrid strokeDasharray="2 3" stroke="rgba(28,24,18,0.12)" vertical={false} />
        <XAxis
          dataKey="name"
          tickLine={false}
          axisLine={{ stroke: "rgba(28,24,18,0.35)" }}
          tick={{ fontSize: 10.5, fontFamily: mono, fill: "#6f6757" }}
        />
        <YAxis
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: 10.5, fontFamily: mono, fill: "#a89d88" }}
          tickFormatter={(v: number) => `${v}`}
          label={{
            value: "accuracy, %",
            angle: -90,
            position: "insideLeft",
            style: { textAnchor: "middle", fill: "#6f6757", fontSize: 10, fontFamily: mono, letterSpacing: "0.05em" },
          }}
        />
        <Tooltip content={<Tip />} cursor={{ fill: "rgba(28,24,18,0.045)" }} />
        {/* stacks: lost sits below retained, so total bar height = region's best */}
        <Bar dataKey="lost" stackId="a" fill={pale} isAnimationActive animationDuration={500} />
        <Bar dataKey="retained" stackId="a" fill={color} radius={[2, 2, 0, 0]} isAnimationActive animationDuration={500} />
        <Bar dataKey="never" fill="rgba(255,255,255,0)" stroke="#a89d88" strokeDasharray="3 3" strokeWidth={1} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}