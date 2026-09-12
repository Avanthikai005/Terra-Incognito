import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { METHODS, averageForgetting, type MethodId, type Results } from "../data/results";

interface ForgettingCurveProps {
  results: Results;
  method: MethodId;
}

interface TipRowProps {
  active?: boolean;
  payload?: ReadonlyArray<{ dataKey: string; value: number | null }>;
}

function Tip({ active, payload }: TipRowProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <div className="tip-title">after the task shown below</div>
      {payload.map((p) => {
        const meta = METHODS.find((m) => m.id === p.dataKey);
        if (p.value === null || p.value === undefined || !meta) return null;
        return (
          <div key={p.dataKey}>
            <span style={{ color: meta.color }}>{meta.label}</span> —{" "}
            {p.value.toFixed(1)} pts
          </div>
        );
      })}
    </div>
  );
}

/**
 * Classic continual-learning visual: average forgetting accumulated per
 * method, plotted across tasks. The gap between the Naive line and the
 * Replay line *is* the contribution of the method.
 */
export default function ForgettingCurve({ results, method }: ForgettingCurveProps) {
  const data = [0, 1, 2].map((t) => {
    const pt: Record<string, number | null> = { t: t + 1 };
    for (const m of METHODS) {
      pt[m.id] = averageForgetting(results.matrices[m.id], t);
    }
    return pt;
  });

  const maxVal = Math.max(1, ...data.map((d) => Math.max(0, ...METHODS.map((m) => d[m.id] ?? 0))));

  const mono = "ui-monospace, SF Mono, Menlo, Consolas, monospace";

  return (
    <div>
      <ResponsiveContainer width="100%" height={210}>
        <LineChart data={data} margin={{ top: 12, right: 14, left: 2, bottom: 2 }}>
          <CartesianGrid strokeDasharray="2 3" stroke="rgba(28,24,18,0.12)" vertical={false} />
          <XAxis
            dataKey="t"
            tickLine={false}
            axisLine={{ stroke: "rgba(28,24,18,0.35)" }}
            tick={{ fontSize: 10.5, fontFamily: mono, fill: "#6f6757" }}
            label={{
              value: "task",
              position: "insideBottom",
              offset: -4,
              style: { fill: "#6f6757", fontSize: 10, fontFamily: mono },
            }}
          />
          <YAxis
            domain={[0, Math.ceil(maxVal + 2)]}
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 10.5, fontFamily: mono, fill: "#a89d88" }}
            tickFormatter={(v: number) => `${v}`}
            label={{
              value: "avg forgetting, pts",
              angle: -90,
              position: "insideLeft",
              style: { textAnchor: "middle", fill: "#6f6757", fontSize: 10, fontFamily: mono },
            }}
          />
          <Tooltip content={<Tip />} cursor={{ stroke: "rgba(28,24,18,0.25)" }} />
          {METHODS.map((m) => {
            const active = m.id === method;
            return (
              <Line
                key={m.id}
                type="monotone"
                dataKey={m.id}
                stroke={m.color}
                strokeWidth={active ? 2.6 : 1.4}
                strokeOpacity={active ? 1 : 0.55}
                dot={{ r: active ? 3 : 2, fill: m.color, strokeWidth: 0 }}
                activeDot={{ r: 4 }}
                connectNulls={false}
                isAnimationActive
                animationDuration={500}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
      <div className="chart-legend">
        {METHODS.map((m) => (
          <span key={m.id}>
            <i style={{ background: m.color, height: 2, borderRadius: 0 }} />
            {m.label}
          </span>
        ))}
      </div>
    </div>
  );
}