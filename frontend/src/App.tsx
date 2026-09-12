import { useEffect, useState } from "react";
import Sidebar, { type PageId } from "./components/Sidebar";
import OverviewPage from "./components/OverviewPage";
import SimulatorPage from "./components/SimulatorPage";
import DomainDistancePage from "./components/DomainDistancePage";
import QualitativePage from "./components/QualitativePage";
import ModelInferencePage from "./components/ModelInferencePage";
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

  // Start on bundled demo data, then upgrade if real results are found
  const [results, setResults] = useState<Results>(() =>
    buildResults(FALLBACK_MATRICES)
  );
  const [qualitative, setQualitative] = useState<QualitativeExamples>(
    QUALITATIVE_FALLBACK
  );

  useEffect(() => {
    let alive = true;

    // Try fetching live API results first
    fetch("/api/results")
      .then((res) => (res.ok ? res.json() : null))
      .then((apiData) => {
        if (!alive || !apiData?.matrices) return;
        const validModes: MethodId[] = ["naive", "joint", "cl"];
        const hasAll = validModes.every((m) => Array.isArray(apiData.matrices[m]));
        if (hasAll) {
          const formattedMatrices: Record<MethodId, (number | null)[][]> = {
            naive: apiData.matrices.naive,
            joint: apiData.matrices.joint,
            cl: apiData.matrices.cl,
          };
          setResults(buildResults(formattedMatrices));
          return;
        }
      })
      .catch(() => {
        // API offline; fall back to static files
      });

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
        {page === "inference" && <ModelInferencePage />}
      </main>
    </div>
  );
}