import {
  Aperture,
  ArrowsClockwise,
  Eye,
  FileMagnifyingGlass,
  Fingerprint,
  Gauge,
  ShieldWarning,
} from "@phosphor-icons/react/dist/ssr";

const steps = [
  {
    number: "01",
    icon: Aperture,
    title: "Decode and normalize",
    text: "The uploaded file is decoded to RGB and transformed by the selected model's own inference pipeline.",
  },
  {
    number: "02",
    icon: Fingerprint,
    title: "Score visual evidence",
    text: "The model looks for learned visual patterns associated with natural photographs and generated imagery.",
  },
  {
    number: "03",
    icon: Gauge,
    title: "Calibrate the verdict",
    text: "The served threshold turns the score into a likely real or likely generated result, with uncertain cases flagged for review.",
  },
  {
    number: "04",
    icon: FileMagnifyingGlass,
    title: "Expose supporting context",
    text: "Provenance, FFT, ELA, and model-specific views remain visible as supporting evidence without secretly changing the verdict.",
  },
];

export function Method() {
  return (
    <section className="mx-auto min-h-[calc(100vh-8rem)] max-w-7xl px-4 py-12 sm:px-6 sm:py-16 lg:px-8">
      <div className="grid gap-10 lg:grid-cols-[0.78fr_1.22fr] lg:gap-16">
        <div>
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-[11px] font-extrabold uppercase tracking-[0.14em] text-muted">
            <Eye size={14} weight="fill" className="text-accent" />
            How it works
          </div>
          <h1 className="balanced text-[2.45rem] font-extrabold leading-[1.08] tracking-[-0.05em] text-ink sm:text-5xl">A useful signal, with its limits left visible.</h1>
          <p className="mt-5 max-w-xl text-base leading-7 text-muted">TruthLens uses the same prediction functions as the command-line evaluation path. The interface adds interpretation and provenance context, not a second scoring system.</p>

          <div className="mt-8 rounded-xl border border-warning/25 bg-warning-soft p-5 text-warning">
            <div className="flex items-center gap-2 font-extrabold">
              <ShieldWarning size={20} weight="fill" /> Not proof of origin
            </div>
            <p className="mt-2 text-sm leading-6">A confident classifier can still be wrong. Use source history, content credentials, reverse image search, and human review for high-stakes decisions.</p>
          </div>
        </div>

        <div className="overflow-hidden rounded-xl border border-line bg-surface soft-shadow">
          {steps.map(({ number, icon: Icon, title, text }, index) => (
            <div key={number} className={`grid grid-cols-[auto_1fr] gap-4 p-5 sm:grid-cols-[3rem_auto_1fr] sm:gap-5 sm:p-6 ${index ? "border-t border-line" : ""}`}>
              <span className="hidden pt-1 font-mono text-xs font-bold text-faint sm:block">{number}</span>
              <span className="grid size-10 place-items-center rounded-lg bg-accent-soft text-accent">
                <Icon size={20} weight="bold" />
              </span>
              <div>
                <h2 className="text-base font-extrabold tracking-[-0.02em] text-ink">{title}</h2>
                <p className="mt-1.5 text-sm leading-6 text-muted">{text}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-12 grid gap-5 border-t border-line pt-10 md:grid-cols-3">
        <div>
          <ArrowsClockwise size={20} weight="bold" className="text-accent" />
          <h2 className="mt-3 font-extrabold text-ink">Training is still moving</h2>
          <p className="mt-2 text-sm leading-6 text-muted">The UI reads model and benchmark outputs from the repository, so checkpoints can improve without rebuilding the product flow.</p>
        </div>
        <div>
          <Fingerprint size={20} weight="bold" className="text-accent" />
          <h2 className="mt-3 font-extrabold text-ink">One inference path</h2>
          <p className="mt-2 text-sm leading-6 text-muted">The web app, project app, and evaluation tools share the same prediction module to avoid inconsistent preprocessing.</p>
        </div>
        <div>
          <ShieldWarning size={20} weight="bold" className="text-accent" />
          <h2 className="mt-3 font-extrabold text-ink">Abstention is a feature</h2>
          <p className="mt-2 text-sm leading-6 text-muted">Some checkpoints mark uncertain outputs for manual review instead of presenting every score as a final answer.</p>
        </div>
      </div>
    </section>
  );
}
