import type { CSSProperties } from "react";
import { REGIONS } from "../data/results";

interface TaskStripProps {
  taskIdx: number; // tasks already trained (0..2)
  matrix: (number | null)[][];
  accent: string;
}

export default function TaskStrip({ taskIdx, matrix, accent }: TaskStripProps) {
  return (
    <div className="tasksheet">
      <div className="task-track">
        {REGIONS.map((region, i) => {
          const done = i <= taskIdx;
          const upNext = i === taskIdx + 1;
          const acc = done ? matrix[taskIdx]?.[i] : null;
          return (
            <div style={{ display: "contents" }} key={region.id}>
              {i > 0 && (
                <span className={`task-connector ${taskIdx > i - 1 ? "on" : ""}`} />
              )}
              <div
                className={`task-card ${done ? "done" : ""} ${upNext ? "next" : ""}`}
                style={{ "--task-color": done ? accent : undefined } as CSSProperties}
              >
                <div className="task-hex">TASK 0{i + 1}</div>
                <div className="task-name">{region.label}</div>
                <div className="task-place">{region.desc}</div>
                <div className="task-status">
                  {done ? (
                    <span className="ok">
                      trained{" "}
                      {acc !== null && acc !== undefined && (
                        <span className="task-acc">· {Math.round(acc)}%</span>
                      )}
                    </span>
                  ) : upNext ? (
                    <span className="warn">up next — press next task</span>
                  ) : (
                    <span className="idle">pending</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}