import { useEffect, useState } from "react";
import Sidebar, { type PageId } from "./components/Sidebar";
import OverviewPage from "./components/OverviewPage";
import SimulatorPage from "./components/SimulatorPage";
import DomainDistancePage from "./components/DomainDistancePage";
import QualitativePage from "./components/QualitativePage";
import {
  FALLBACK_MATRICES,
  buildResults,
  loadResults,
  type MethodId,
  type Results,
} from "./data/results";
import {
  QUALITATIVE_FALLBACK,
  loadQualitative,
  type QualitativeExamples,
} from "./data/qualitative";

export default function App() {
  const [page, setPage] = useState<PageId>("overview");
  const [method, setMethod] = useState<MethodId>("naive");

  // Start on bundled demo data, then silently upgrade if a real results.json /
  // qualitative.json is present (see src/data/results.ts header comments).
  const [results, setResults] = useState<Results>(() =>
    buildResults(FALLBACK_MATRICES)
  );
  const [qualitative, setQualitative] = useState<QualitativeExamples>(
    QUALITATIVE_FALLBACK
  );

  useEffect(() => {
    let alive = true;
    loadResults().then((r) => alive && setResults(r));
    loadQualitative().then((q) => alive && setQualitative(q));
    return () => {
      alive = false;
    };
  }, []);

  const changeMethod = (m: MethodId) => setMethod(m);

  return (
    <div className="app">
      <Sidebar page={page} onNavigate={setPage} method={method} />
      <main className="main">
        {page === "overview" && <OverviewPage results={results} onStart={() => setPage("simulator")} />}
        {page === "simulator" && (
          <SimulatorPage results={results} method={method} onMethodChange={changeMethod} />
        )}
        {page === "distance" && <DomainDistancePage />}
        {page === "qualitative" && (
          <QualitativePage
            qualitative={qualitative}
            method={method}
            onMethodChange={changeMethod}
          />
        )}
      </main>
    </div>
  );
}