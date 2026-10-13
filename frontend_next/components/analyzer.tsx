"use client";

import {
  ArrowRight,
  CaretDown,
  Check,
  DownloadSimple,
  FileImage,
  Lightning,
  SpinnerGap,
  Trash,
  UploadSimple,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import Image from "next/image";
import { ChangeEvent, DragEvent, useRef, useState } from "react";
import { AnalysisResult } from "@/components/analysis-result";
import { MODEL_GROUPS, MODELS, SAMPLES } from "@/lib/models";
import type { QueueItem } from "@/lib/types";

const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp"]);
const MAX_BYTES = 25 * 1024 * 1024;

function uniqueId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function exportCsv(items: QueueItem[]) {
  const rows = items.flatMap((item) => item.response ? [[
    item.response.filename,
    item.response.model_type,
    item.response.result.prediction,
    item.response.result.confidence,
    item.response.result.probabilities.REAL,
    item.response.result.probabilities.FAKE,
    Boolean(item.response.result.review_recommended),
  ]] : []);
  const escape = (value: unknown) => `"${String(value).replaceAll('"', '""')}"`;
  const csv = [
    ["filename", "model", "prediction", "confidence", "probability_real", "probability_fake", "review_recommended"],
    ...rows,
  ].map((row) => row.map(escape).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "truthlens-results.csv";
  anchor.click();
  URL.revokeObjectURL(url);
}

export function Analyzer() {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [modelKey, setModelKey] = useState("swin_improved_ep6");
  const [withGradcam, setWithGradcam] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [sampleLoading, setSampleLoading] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const selectedModel = MODELS.find((model) => model.key === modelKey) ?? MODELS[0];
  const completed = items.filter((item) => item.status === "complete").length;
  const activeItem = items.find((item) => item.id === activeId) ?? items.find((item) => item.response);

  function addFiles(incoming: File[]) {
    setNotice(null);
    const accepted: File[] = [];
    let rejected = 0;

    for (const file of incoming) {
      if (!ACCEPTED_TYPES.has(file.type) || file.size > MAX_BYTES) {
        rejected += 1;
        continue;
      }
      accepted.push(file);
    }

    const nextItems = accepted.map<QueueItem>((file) => ({
      id: uniqueId(),
      file,
      preview: URL.createObjectURL(file),
      status: "ready",
    }));
    setItems((current) => [...current, ...nextItems]);
    if (nextItems.length && !activeId) setActiveId(nextItems[0].id);
    if (rejected) {
      setNotice("Some files were skipped. Use supported images no larger than 25 MB each.");
    }
  }

  function handleInput(event: ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    addFiles(Array.from(event.dataTransfer.files));
  }

  function updateItem(id: string, change: Partial<QueueItem>) {
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...change } : item)));
  }

  function removeItem(id: string) {
    const removed = items.find((item) => item.id === id);
    if (removed) URL.revokeObjectURL(removed.preview);
    const remaining = items.filter((item) => item.id !== id);
    setItems(remaining);
    if (activeId === id) setActiveId(remaining[0]?.id ?? null);
  }

  function clearQueue() {
    items.forEach((item) => URL.revokeObjectURL(item.preview));
    setItems([]);
    setActiveId(null);
    setNotice(null);
  }

  function changeModel(nextModel: string) {
    setModelKey(nextModel);
    setWithGradcam(false);
    setItems((current) =>
      current.map((item) => ({ ...item, status: "ready", response: undefined, error: undefined })),
    );
  }

  async function addSample(src: string, name: string) {
    setSampleLoading(src);
    try {
      const response = await fetch(src);
      if (!response.ok) throw new Error("Sample could not be loaded.");
      const blob = await response.blob();
      addFiles([new File([blob], `${name.toLowerCase().replaceAll(" ", "-")}.jpg`, { type: "image/jpeg" })]);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Sample could not be loaded.");
    } finally {
      setSampleLoading(null);
    }
  }

  async function analyzeOne(item: QueueItem) {
    updateItem(item.id, { status: "analyzing", response: undefined, error: undefined });
    setActiveId(item.id);
    const formData = new FormData();
    formData.append("file", item.file);
    formData.append("model_type", modelKey);
    formData.append("with_gradcam", String(withGradcam));

    try {
      const response = await fetch("/api/analyze", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analysis failed.");
      updateItem(item.id, { status: "complete", response: data });
    } catch (error) {
      updateItem(item.id, {
        status: "error",
        error: error instanceof Error ? error.message : "Analysis failed.",
      });
    }
  }

  async function analyzeQueue() {
    if (!items.length || isRunning) return;
    setIsRunning(true);
    let targets = items.filter((item) => item.status === "ready" || item.status === "error");
    if (!targets.length) targets = items;
    for (const item of targets) await analyzeOne(item);
    setIsRunning(false);
  }

  return (
    <>
      <section className="mx-auto max-w-7xl px-4 pb-12 pt-10 sm:px-6 sm:pt-14 lg:px-8 lg:pb-16">
        <div className="grid items-start gap-9 lg:grid-cols-[0.82fr_1.18fr] lg:gap-14">
          <div className="pt-1 lg:pt-8">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-[11px] font-extrabold uppercase tracking-[0.14em] text-muted">
              <span className="size-1.5 rounded-full bg-accent" />
              Image authenticity analysis
            </div>
            <h1 className="balanced max-w-xl text-[2.55rem] font-extrabold leading-[1.04] tracking-[-0.055em] text-ink sm:text-[3.35rem]">
              Check an image before you trust it.
            </h1>
            <p className="mt-5 max-w-xl text-base leading-7 text-muted sm:text-lg">
              Upload one image or a small batch. TruthLens checks each file with your selected project model and keeps the supporting signals visible.
            </p>

            <div className="mt-8 border-y border-line py-5">
              <label htmlFor="model-select" className="mb-2 block text-xs font-extrabold uppercase tracking-[0.12em] text-muted">
                Detection model
              </label>
              <div className="relative">
                <select
                  id="model-select"
                  value={modelKey}
                  onChange={(event) => changeModel(event.target.value)}
                  disabled={isRunning}
                  className="h-11 w-full appearance-none rounded-lg border border-line-strong bg-surface px-3 pr-10 text-sm font-bold text-ink transition hover:border-accent focus:border-accent"
                >
                  {MODEL_GROUPS.map((group) => (
                    <optgroup key={group} label={group}>
                      {MODELS.filter((model) => model.group === group).map((model) => (
                        <option key={model.key} value={model.key}>{model.label}</option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <CaretDown size={16} weight="bold" className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-muted" />
              </div>
              <p className="mt-2 text-sm leading-6 text-muted">{selectedModel.description}</p>

              {selectedModel.supportsGradcam && (
                <label className="mt-4 flex items-center justify-between gap-4 rounded-lg bg-surface-subtle px-3.5 py-3">
                  <span>
                    <span className="block text-sm font-bold text-ink">Generate Grad-CAM</span>
                    <span className="mt-0.5 block text-xs text-muted">Adds a slower attention-map pass.</span>
                  </span>
                  <input
                    type="checkbox"
                    checked={withGradcam}
                    onChange={(event) => setWithGradcam(event.target.checked)}
                    className="size-4 accent-accent"
                  />
                </label>
              )}
            </div>

            <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-xs font-semibold text-muted">
              <span className="flex items-center gap-1.5"><Check size={15} weight="bold" className="text-real" /> JPG, PNG, WebP, BMP</span>
              <span className="flex items-center gap-1.5"><Check size={15} weight="bold" className="text-real" /> Up to 25 MB each</span>
              <span className="flex items-center gap-1.5"><Check size={15} weight="bold" className="text-real" /> No fixed batch limit</span>
            </div>
          </div>

          <div className="overflow-hidden rounded-xl border border-line bg-surface soft-shadow">
            <div className="relative h-44 overflow-hidden border-b border-line sm:h-52">
              <Image
                src="/images/authenticity-review.png"
                alt="Two orchid photographs beside a photographer's loupe, one with subtle digital artifacts"
                fill
                priority
                className="object-cover object-center"
                sizes="(max-width: 1024px) 100vw, 58vw"
              />
              <div className="absolute left-4 top-4 rounded-md bg-surface/90 px-2.5 py-1.5 text-[10px] font-extrabold uppercase tracking-[0.12em] text-ink backdrop-blur">
                Inspect the image, not the story
              </div>
            </div>

            <div className="p-4 sm:p-5">
              <div
                onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`flex min-h-52 flex-col items-center justify-center rounded-lg border border-dashed px-5 py-8 text-center transition ${
                  isDragging ? "border-accent bg-accent-soft" : "border-line-strong bg-surface-raised hover:border-accent"
                }`}
              >
                <span className="grid size-11 place-items-center rounded-lg bg-accent-soft text-accent">
                  <UploadSimple size={22} weight="bold" />
                </span>
                <p className="mt-4 text-base font-extrabold tracking-[-0.02em] text-ink">Drop images here</p>
                <p className="mt-1 text-sm text-muted">or choose them from your device</p>
                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  disabled={isRunning}
                  className="mt-5 inline-flex h-10 items-center gap-2 rounded-lg bg-ink px-4 text-sm font-bold text-surface transition hover:opacity-90 disabled:opacity-45"
                >
                  <FileImage size={18} weight="bold" />
                  Choose images
                </button>
                <input
                  ref={inputRef}
                  type="file"
                  multiple
                  accept="image/jpeg,image/png,image/webp,image/bmp"
                  onChange={handleInput}
                  className="sr-only"
                />
              </div>
            </div>
          </div>
        </div>

        {notice && (
          <div className="mt-5 flex items-start gap-2.5 rounded-lg border border-warning/25 bg-warning-soft px-4 py-3 text-sm text-warning">
            <WarningCircle size={18} weight="fill" className="mt-0.5 shrink-0" />
            <span>{notice}</span>
            <button type="button" onClick={() => setNotice(null)} className="ml-auto" aria-label="Dismiss notice" title="Dismiss">
              <X size={17} weight="bold" />
            </button>
          </div>
        )}

        {!items.length && (
          <div className="mt-10">
            <div className="mb-4 flex items-end justify-between gap-4">
              <div>
                <p className="text-xs font-extrabold uppercase tracking-[0.12em] text-muted">Quick test</p>
                <h2 className="mt-1 text-xl font-extrabold tracking-[-0.035em] text-ink">Try a held-out sample</h2>
              </div>
              <p className="hidden text-xs text-muted sm:block">Not used for training</p>
            </div>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              {SAMPLES.map((sample) => (
                <button
                  key={sample.src}
                  type="button"
                  onClick={() => addSample(sample.src, sample.name)}
                  disabled={sampleLoading !== null}
                  className="group overflow-hidden rounded-lg border border-line bg-surface text-left transition hover:-translate-y-0.5 hover:border-line-strong hover:shadow-md disabled:opacity-60"
                >
                  <span className="relative block aspect-[1.35] overflow-hidden bg-surface-subtle">
                    <Image src={sample.src} alt={sample.name} fill className="object-cover transition duration-300 group-hover:scale-[1.025]" sizes="(max-width: 1024px) 50vw, 25vw" />
                    {sampleLoading === sample.src && (
                      <span className="absolute inset-0 grid place-items-center bg-surface/75">
                        <SpinnerGap size={22} weight="bold" className="animate-spin text-accent" />
                      </span>
                    )}
                  </span>
                  <span className="flex items-center justify-between gap-2 px-3 py-2.5">
                    <span className="truncate text-xs font-bold text-ink">{sample.name}</span>
                    <span className={`text-[10px] font-extrabold ${sample.truth === "REAL" ? "text-real" : "text-fake"}`}>{sample.truth}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      {items.length > 0 && (
        <section className="border-y border-line bg-surface">
          <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
            <div className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="text-xs font-extrabold uppercase tracking-[0.12em] text-muted">Analysis queue</p>
                <h2 className="mt-1 text-2xl font-extrabold tracking-[-0.04em] text-ink">
                  {completed ? `${completed} of ${items.length} complete` : `${items.length} ${items.length === 1 ? "image" : "images"} ready`}
                </h2>
              </div>
              <div className="flex items-center gap-2">
                {completed > 0 && (
                  <button
                    type="button"
                    onClick={() => exportCsv(items)}
                    disabled={isRunning}
                    className="hidden h-10 items-center gap-2 rounded-lg border border-line bg-surface px-3.5 text-sm font-bold text-muted transition hover:border-line-strong hover:text-ink disabled:opacity-40 sm:inline-flex"
                  >
                    <DownloadSimple size={17} weight="bold" /> Export CSV
                  </button>
                )}
                <button
                  type="button"
                  onClick={clearQueue}
                  disabled={isRunning}
                  className="inline-flex h-10 items-center gap-2 rounded-lg border border-line bg-surface px-3.5 text-sm font-bold text-muted transition hover:border-line-strong hover:text-ink disabled:opacity-40"
                >
                  <Trash size={17} weight="bold" /> Clear
                </button>
                <button
                  type="button"
                  onClick={analyzeQueue}
                  disabled={isRunning}
                  className="inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-extrabold text-white transition hover:bg-accent-strong disabled:opacity-55"
                >
                  {isRunning ? <SpinnerGap size={18} weight="bold" className="animate-spin" /> : <Lightning size={18} weight="fill" />}
                  {isRunning ? "Analyzing" : completed === items.length ? "Run again" : "Analyze queue"}
                  {!isRunning && <ArrowRight size={17} weight="bold" />}
                </button>
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {items.map((item) => (
                <div
                  key={item.id}
                  className={`group flex min-w-0 items-center rounded-lg border p-2 text-left transition ${
                    activeItem?.id === item.id ? "border-accent bg-accent-soft" : "border-line bg-surface-raised hover:border-line-strong"
                  }`}
                >
                  <button type="button" onClick={() => setActiveId(item.id)} className="flex min-w-0 flex-1 items-center gap-3 text-left">
                    <span className="relative size-12 shrink-0 overflow-hidden rounded-md bg-surface-subtle">
                      <Image src={item.preview} alt="" fill unoptimized className="object-cover" sizes="48px" />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs font-bold text-ink">{item.file.name}</span>
                      <span className="mt-1 flex items-center gap-1.5 text-[11px] font-semibold text-muted">
                        {item.status === "analyzing" && <SpinnerGap size={13} className="animate-spin text-accent" />}
                        {item.status === "complete" && <Check size={13} weight="bold" className="text-real" />}
                        {item.status === "error" && <WarningCircle size={13} weight="fill" className="text-fake" />}
                        {item.status === "ready" ? formatBytes(item.file.size) : item.status}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => removeItem(item.id)}
                    disabled={isRunning}
                    className="grid size-7 shrink-0 place-items-center rounded-md text-faint opacity-60 transition hover:bg-surface hover:text-ink group-hover:opacity-100 disabled:hidden"
                    aria-label={`Remove ${item.file.name}`}
                    title="Remove"
                  >
                    <X size={15} weight="bold" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {activeItem && (
        <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
          <AnalysisResult item={activeItem} />
        </section>
      )}
    </>
  );
}
