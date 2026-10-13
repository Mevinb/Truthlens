"use client";

import { ChartBar, Info, SpinnerGap, TrendUp, WarningCircle } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import type { BenchmarkRow, ResolutionRow } from "@/lib/types";

function percent(value: number) {
  return `${value.toFixed(2)}%`;
}

function ratioPercent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

export function Benchmarks() {
  const [models, setModels] = useState<BenchmarkRow[]>([]);
  const [resolutions, setResolutions] = useState<ResolutionRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/benchmarks")
      .then(async (response) => {
        if (!response.ok) throw new Error("offline");
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        setModels(data.models ?? []);
        setResolutions(data.resolutions ?? []);
      })
      .catch(() => { if (!cancelled) setOffline(true); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const leader = models[0];

  return (
    <section className="mx-auto min-h-[calc(100vh-8rem)] max-w-7xl px-4 py-12 sm:px-6 sm:py-16 lg:px-8">
      <div className="max-w-3xl">
        <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-[11px] font-extrabold uppercase tracking-[0.14em] text-muted">
          <ChartBar size={14} weight="fill" className="text-accent" />
          Measured performance
        </div>
        <h1 className="balanced text-[2.45rem] font-extrabold leading-[1.08] tracking-[-0.05em] text-ink sm:text-5xl">See where the detector holds up, and where it does not.</h1>
        <p className="mt-5 max-w-2xl text-base leading-7 text-muted">These values come from the metrics files already produced by this repository. Re-running the benchmark updates this view through the Python API.</p>
      </div>

      {loading && (
        <div className="mt-10 flex items-center gap-3 rounded-xl border border-line bg-surface p-6">
          <SpinnerGap size={20} weight="bold" className="animate-spin text-accent" />
          <p className="text-sm font-bold text-muted">Loading measured results</p>
        </div>
      )}

      {offline && (
        <div className="mt-10 flex max-w-2xl items-start gap-3 rounded-xl border border-warning/25 bg-warning-soft p-5 text-warning">
          <WarningCircle size={21} weight="fill" className="mt-0.5 shrink-0" />
          <div>
            <p className="font-extrabold">Benchmark service is offline</p>
            <p className="mt-1 text-sm leading-6">Start this frontend with <code className="rounded bg-surface px-1.5 py-0.5 font-mono text-xs text-ink">./run.sh</code> to load the repository metrics.</p>
          </div>
        </div>
      )}

      {!loading && leader && (
        <>
          <div className="mt-10 overflow-hidden rounded-xl border border-line bg-surface soft-shadow">
            <div className="grid md:grid-cols-[1.25fr_0.75fr]">
              <div className="p-6 sm:p-8">
                <div className="flex items-center gap-2 text-xs font-extrabold uppercase tracking-[0.12em] text-real">
                  <TrendUp size={16} weight="bold" /> Current test split leader
                </div>
                <h2 className="mt-3 text-2xl font-extrabold tracking-[-0.04em] text-ink">{leader.Model}</h2>
                <p className="mt-3 max-w-xl text-sm leading-6 text-muted">A strong score on this split is encouraging, but it does not guarantee performance on unseen generators, aggressive recompression, or screenshots.</p>
              </div>
              <div className="grid grid-cols-3 border-t border-line bg-surface-raised md:grid-cols-1 md:border-l md:border-t-0">
                <div className="p-4 md:border-b md:p-5">
                  <p className="text-[10px] font-extrabold uppercase tracking-[0.12em] text-muted">Accuracy</p>
                  <p className="mt-1 font-mono text-xl font-bold text-ink">{percent(leader.Accuracy)}</p>
                </div>
                <div className="border-l border-line p-4 md:border-b md:border-l-0 md:p-5">
                  <p className="text-[10px] font-extrabold uppercase tracking-[0.12em] text-muted">AUC</p>
                  <p className="mt-1 font-mono text-xl font-bold text-ink">{ratioPercent(leader.AUC)}</p>
                </div>
                <div className="border-l border-line p-4 md:border-l-0 md:p-5">
                  <p className="text-[10px] font-extrabold uppercase tracking-[0.12em] text-muted">Images</p>
                  <p className="mt-1 font-mono text-xl font-bold text-ink">{leader.Images?.toLocaleString() ?? "N/A"}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 overflow-hidden rounded-xl border border-line bg-surface">
            <div className="border-b border-line px-5 py-4 sm:px-6">
              <h2 className="text-lg font-extrabold tracking-[-0.03em] text-ink">Model comparison</h2>
              <p className="mt-1 text-sm text-muted">Sorted by balanced test split accuracy.</p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] border-collapse text-left text-sm">
                <thead className="bg-surface-raised text-[10px] font-extrabold uppercase tracking-[0.12em] text-muted">
                  <tr>
                    <th className="px-6 py-3">Model</th>
                    <th className="px-4 py-3">Accuracy</th>
                    <th className="px-4 py-3">F1</th>
                    <th className="px-4 py-3">AUC</th>
                    <th className="px-4 py-3">Real acc</th>
                    <th className="px-4 py-3">Fake acc</th>
                    <th className="px-6 py-3 text-right">Images</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((row, index) => (
                    <tr key={row.Model} className="border-t border-line transition hover:bg-surface-raised">
                      <td className="px-6 py-4 font-bold text-ink">
                        <span className="mr-3 inline-block w-5 font-mono text-xs text-faint">{String(index + 1).padStart(2, "0")}</span>
                        {row.Model}
                      </td>
                      <td className="px-4 py-4 font-mono text-xs font-bold text-ink">{percent(row.Accuracy)}</td>
                      <td className="px-4 py-4 font-mono text-xs text-muted">{ratioPercent(row.F1)}</td>
                      <td className="px-4 py-4 font-mono text-xs text-muted">{ratioPercent(row.AUC)}</td>
                      <td className="px-4 py-4 font-mono text-xs text-muted">{row["Real acc"] === undefined ? "N/A" : percent(row["Real acc"])}</td>
                      <td className="px-4 py-4 font-mono text-xs text-muted">{row["Fake acc"] === undefined ? "N/A" : percent(row["Fake acc"])}</td>
                      <td className="px-6 py-4 text-right font-mono text-xs text-muted">{row.Images?.toLocaleString() ?? "N/A"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {resolutions.length > 0 && (
            <div className="mt-5 grid gap-5 lg:grid-cols-[0.75fr_1.25fr]">
              <div className="rounded-xl border border-line bg-surface p-5 sm:p-6">
                <div className="flex items-center gap-2 text-accent">
                  <Info size={18} weight="fill" />
                  <h2 className="text-base font-extrabold text-ink">Resolution matters</h2>
                </div>
                <p className="mt-3 text-sm leading-6 text-muted">The blended headline can hide a weaker image regime. Most everyday uploads land in the high-resolution tier, so inspect its row separately.</p>
              </div>
              <div className="rounded-xl border border-line bg-surface p-5 sm:p-6">
                <div className="space-y-5">
                  {resolutions.map((row) => (
                    <div key={row["Resolution tier"]}>
                      <div className="mb-2 flex items-center justify-between gap-4 text-xs">
                        <span className="font-bold text-ink">{row["Resolution tier"]}</span>
                        <span className="font-mono font-bold text-muted">{percent(row.Accuracy)}</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-surface-subtle">
                        <div className="h-full rounded-full bg-accent" style={{ width: `${row.Accuracy}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
