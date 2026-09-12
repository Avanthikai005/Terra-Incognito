import { METHODS, type MethodId } from "../data/results";

export type PageId = "overview" | "simulator" | "distance" | "qualitative";

const NAV: { id: PageId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "simulator", label: "Simulator" },
  { id: "distance", label: "Domain distance" },
  { id: "qualitative", label: "Qualitative plates" },
];

interface SidebarProps {
  page: PageId;
  onNavigate: (page: PageId) => void;
  method: MethodId;
}

export default function Sidebar({ page, onNavigate, method }: SidebarProps) {
  const active = METHODS.find((m) => m.id === method)!;
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">TI</span>
        <div>
          <div className="brand-title">Terra Incognita</div>
          <div className="brand-sub">continual learning · disaster response</div>
        </div>
      </div>

      <div className="contents-label">Contents</div>
      <nav className="nav">
        {NAV.map((item, i) => (
          <button
            key={item.id}
            className={`nav-item ${page === item.id ? "active" : ""}`}
            onClick={() => onNavigate(item.id)}
          >
            <span className="nav-num">0{i + 1}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="contents-label">Active regime</div>
      <div className="regime-readout">
        <span className="rr-dot" style={{ background: active.color }} />
        <span className="rr-main">{active.label}</span>
        <span className="rr-sub">{active.tagline}</span>
      </div>

      <div className="sidebar-foot">
        build v1.0 · demo build · data precomputed
        <br />
        seed 42 · xbd / xview2
      </div>
    </aside>
  );
}