"use client";

import {
  CheckCircle,
  DownloadSimple,
  Eye,
  FileText,
  ImageSquare,
  Info,
  MagnifyingGlass,
  SpinnerGap,
  WarningCircle,
  WarningOctagon,
} from "@phosphor-icons/react";
import Image from "next/image";
import { useMemo, useState } from "react";
import { MODELS } from "@/lib/models";
import type { QueueItem } from "@/lib/types";

type ResultTab = "summary" | "signals" | "provenance" | "heatmap";

function readableLabel(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function cleanText(value: string) {
  return value.replace(/[–—]/g, " - ");
}

function readableValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Not available";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.map(readableValue).join(", ");
  if (typeof value === "object") return cleanText(JSON.stringify(value));
  return cleanText(String(value));
}

function objectRows(value?: Record<string, unknown>) {
  if (!value) return [];
  return Object.entries(value).flatMap(([key, item]) => {
    if (item && typeof item === "object" && !Array.isArray(item)) {
      return Object.entries(item as Record<string, unknown>).map(([nestedKey, nestedValue]) => ({
        label: `${readableLabel(key)} · ${readableLabel(nestedKey)}`,
        value: readableValue(nestedValue),
      }));
    }
    return [{ label: readableLabel(key), value: readableValue(item) }];
  });
}

function downloadResult(filename: string, result: unknown) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${filename.replace(/\.[^.]+$/, "")}-truthlens.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function EmptyState({ status, error }: { status: QueueItem["status"]; error?: string }) {
  if (status === "analyzing") {
    return (
      <div className="rounded-xl border border-line bg-surface p-6 soft-shadow">
        <div className="flex items-center gap-3">
          <SpinnerGap size={22} weight="bold" className="animate-spin text-accent" />
          <div>
            <p className="font-extrabold text-ink">Running the detector</p>
            <p className="mt-0.5 text-sm text-muted">The first image can take longer while the model loads.</p>
          </div>
        </div>
        <div className="mt-6 space-y-3">
          <div className="h-20 animate-pulse rounded-lg bg-surface-subtle" />
          <div className="h-3 w-4/5 animate-pulse rounded bg-surface-subtle" />
          <div className="h-3 w-3/5 animate-pulse rounded bg-surface-subtle" />
        </div>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="rounded-xl border border-fake/25 bg-fake-soft p-6">
        <WarningCircle size={24} weight="fill" className="text-fake" />
        <h3 className="mt-4 text-lg font-extrabold text-ink">Analysis could not finish</h3>
        <p className="mt-2 max-w-xl text-sm leading-6 text-muted">{error || "Check the model service and try again."}</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-dashed border-line-strong bg-surface p-6 text-center">
      <MagnifyingGlass size={24} weight="bold" className="mx-auto text-accent" />
      <h3 className="mt-3 font-extrabold text-ink">Ready for analysis</h3>
      <p className="mt-1 text-sm text-muted">Choose Analyze queue to run the selected model.</p>
    </div>
  );
}

export function AnalysisResult({ item }: { item: QueueItem }) {
  const [activeTab, setActiveTab] = useState<ResultTab>("summary");
  const response = item.response;
  const result = response?.result;
  const signalRows = useMemo(() => [
    ...objectRows(result?.spectral_analysis),
    ...objectRows(result?.ela_metrics),
    ...objectRows(result?.view_probabilities),
  ], [result]);
  const provenanceRows = useMemo(() => objectRows(result?.provenance_analysis), [result]);

  if (!response || !result) return <EmptyState status={item.status} error={item.error} />;

  const visibleTab = activeTab === "heatmap" && !response.heatmap ? "summary" : activeTab;

  const fake = result.prediction === "FAKE";
  const review = Boolean(result.review_recommended);
  const model = MODELS.find((candidate) => candidate.key === response.model_type);
  const tabs: { id: ResultTab; label: string; icon: typeof Info }[] = [
    { id: "summary", label: "Summary", icon: FileText },
    { id: "signals", label: "Signals", icon: Eye },
    { id: "provenance", label: "Provenance", icon: Info },
  ];
  if (response.heatmap) tabs.push({ id: "heatmap", label: "Heatmap", icon: ImageSquare });

  return (
    <div className="animate-[fade-in_240ms_ease-out]">
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-extrabold uppercase tracking-[0.12em] text-muted">Selected result</p>
          <h2 className="mt-1 max-w-2xl truncate text-2xl font-extrabold tracking-[-0.04em] text-ink">{response.filename}</h2>
        </div>
        <button
          type="button"
          onClick={() => downloadResult(response.filename, response)}
          className="inline-flex h-10 items-center justify-center gap-2 self-start rounded-lg border border-line bg-surface px-3.5 text-sm font-bold text-muted transition hover:border-line-strong hover:text-ink sm:self-auto"
        >
          <DownloadSimple size={17} weight="bold" /> Export JSON
        </button>
      </div>

      <div className="grid gap-5 lg:grid-cols-[0.92fr_1.08fr]">
        <div className="overflow-hidden rounded-xl border border-line bg-surface soft-shadow">
          <div className="relative aspect-[4/3] bg-surface-subtle">
            <Image src={item.preview} alt={`Uploaded image ${response.filename}`} fill unoptimized className="object-contain" sizes="(max-width: 1024px) 100vw, 46vw" />
          </div>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line px-4 py-3 text-xs font-semibold text-muted">
            <span>{response.width} × {response.height}px</span>
            <span>{readableLabel(response.resolution_tier)}</span>
            <span>{model?.shortLabel ?? response.model_type}</span>
          </div>
        </div>

        <div className="rounded-xl border border-line bg-surface p-5 soft-shadow sm:p-6">
          <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className={`inline-flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs font-extrabold uppercase tracking-[0.1em] ${fake ? "bg-fake-soft text-fake" : "bg-real-soft text-real"}`}>
                {fake ? <WarningOctagon size={17} weight="fill" /> : <CheckCircle size={17} weight="fill" />}
                {fake ? "Likely AI-generated" : "Likely real"}
              </div>
              <p className="mt-4 text-[2.7rem] font-extrabold leading-none tracking-[-0.06em] text-ink">
                {result.confidence.toFixed(1)}<span className="ml-1 text-xl text-muted">%</span>
              </p>
              <p className="mt-1 text-sm font-semibold text-muted">confidence in this verdict</p>
            </div>
            {review && (
              <div className="max-w-48 rounded-lg border border-warning/25 bg-warning-soft px-3 py-2.5 text-xs leading-5 text-warning">
                <strong className="block font-extrabold">Manual review advised</strong>
                The score or model views are inconclusive.
              </div>
            )}
          </div>

          <div className="mt-7 space-y-4">
            {(["REAL", "FAKE"] as const).map((label) => {
              const value = result.probabilities[label];
              return (
                <div key={label}>
                  <div className="mb-1.5 flex items-center justify-between text-xs font-extrabold">
                    <span className={label === "REAL" ? "text-real" : "text-fake"}>{label}</span>
                    <span className="font-mono text-ink">{value.toFixed(2)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-surface-subtle">
                    <div className={`h-full rounded-full transition-[width] duration-500 ${label === "REAL" ? "bg-real" : "bg-fake"}`} style={{ width: `${Math.max(1, value)}%` }} />
                  </div>
                </div>
              );
            })}
          </div>

          {fake && result.likely_generator && result.likely_generator !== "N/A" && (
            <div className="mt-6 border-t border-line pt-4">
              <p className="text-[11px] font-extrabold uppercase tracking-[0.12em] text-muted">Generator clue</p>
              <p className="mt-1.5 text-sm font-bold text-ink">{cleanText(result.likely_generator)}</p>
            </div>
          )}
        </div>
      </div>

      <div className="mt-5 overflow-hidden rounded-xl border border-line bg-surface">
        <div className="flex gap-1 overflow-x-auto border-b border-line p-2" role="tablist" aria-label="Analysis evidence">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={visibleTab === id}
              onClick={() => setActiveTab(id)}
              className={`inline-flex h-9 shrink-0 items-center gap-2 rounded-lg px-3 text-xs font-extrabold transition ${
                visibleTab === id ? "bg-surface-subtle text-ink" : "text-muted hover:text-ink"
              }`}
            >
              <Icon size={16} weight={visibleTab === id ? "fill" : "bold"} /> {label}
            </button>
          ))}
        </div>

        <div className="p-5 sm:p-6">
          {visibleTab === "summary" && (
            <div className="grid gap-7 lg:grid-cols-[1.1fr_0.9fr]">
              <div>
                <h3 className="text-base font-extrabold tracking-[-0.02em] text-ink">What influenced the result</h3>
                {result.explanation?.length ? (
                  <ul className="mt-4 space-y-3">
                    {result.explanation.map((reason) => (
                      <li key={reason} className="flex gap-3 text-sm leading-6 text-muted">
                        <span className="mt-2 size-1.5 shrink-0 rounded-full bg-accent" />
                        <span>{cleanText(reason)}</span>
                      </li>
                    ))}
                  </ul>
                ) : <p className="mt-3 text-sm text-muted">No explanation was returned for this model.</p>}
              </div>
              <div className="border-t border-line pt-5 lg:border-l lg:border-t-0 lg:pl-7 lg:pt-0">
                <h3 className="text-base font-extrabold tracking-[-0.02em] text-ink">Reading this result</h3>
                <p className="mt-3 text-sm leading-6 text-muted">
                  A probability is the model&apos;s score for this image, not a verified fact about its origin. Compression, editing, screenshots, and unseen generators can change the result.
                </p>
                <div className="mt-4 flex items-start gap-2 rounded-lg bg-surface-subtle px-3.5 py-3 text-xs leading-5 text-muted">
                  <Info size={17} weight="fill" className="mt-0.5 shrink-0 text-accent" />
                  Compare the output with provenance records and source context before making a decision.
                </div>
              </div>
            </div>
          )}

          {visibleTab === "signals" && (
            <div>
              <div className="flex max-w-3xl items-start gap-2.5 rounded-lg bg-surface-subtle px-3.5 py-3 text-xs leading-5 text-muted">
                <Info size={17} weight="fill" className="mt-0.5 shrink-0 text-accent" />
                FFT and ELA are descriptive signals only. They do not move the model verdict.
              </div>
              {signalRows.length ? (
                <dl className="mt-5 grid gap-x-8 sm:grid-cols-2">
                  {signalRows.map((row) => (
                    <div key={row.label} className="flex items-start justify-between gap-5 border-b border-line py-3 text-sm">
                      <dt className="text-muted">{row.label}</dt>
                      <dd className="max-w-[55%] text-right font-mono text-xs font-bold text-ink">{row.value}</dd>
                    </div>
                  ))}
                </dl>
              ) : <p className="mt-5 text-sm text-muted">This model did not return additional forensic signals.</p>}
            </div>
          )}

          {visibleTab === "provenance" && (
            <div>
              <h3 className="text-base font-extrabold tracking-[-0.02em] text-ink">Embedded source information</h3>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">Metadata and content credentials can add context, but missing metadata does not imply an image is synthetic.</p>
              {provenanceRows.length ? (
                <dl className="mt-5 grid gap-x-8 sm:grid-cols-2">
                  {provenanceRows.map((row) => (
                    <div key={row.label} className="flex items-start justify-between gap-5 border-b border-line py-3 text-sm">
                      <dt className="text-muted">{row.label}</dt>
                      <dd className="max-w-[58%] break-words text-right text-xs font-bold text-ink">{row.value}</dd>
                    </div>
                  ))}
                </dl>
              ) : <p className="mt-5 text-sm text-muted">No provenance details were found in this file.</p>}
            </div>
          )}

          {visibleTab === "heatmap" && response.heatmap && (
            <div className="grid gap-5 lg:grid-cols-[minmax(0,680px)_1fr]">
              <div className="relative aspect-[4/3] overflow-hidden rounded-lg bg-surface-subtle">
                <Image src={response.heatmap} alt="Grad-CAM attention heatmap" fill unoptimized className="object-contain" sizes="(max-width: 1024px) 100vw, 680px" />
              </div>
              <div>
                <h3 className="text-base font-extrabold text-ink">Model attention</h3>
                <p className="mt-2 text-sm leading-6 text-muted">Warmer regions contributed more strongly to the CNN activation. This helps inspect model focus, but it is not a pixel-level manipulation map.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
