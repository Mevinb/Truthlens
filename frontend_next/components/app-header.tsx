"use client";

import { ChartBar, Flask, ImageSquare, Scan } from "@phosphor-icons/react";
import type { AppView } from "@/lib/types";
import { ThemeToggle } from "@/components/theme-toggle";

const navigation = [
  { id: "analyze" as const, label: "Analyze", icon: ImageSquare },
  { id: "benchmarks" as const, label: "Benchmarks", icon: ChartBar },
  { id: "method" as const, label: "Method", icon: Flask },
];

interface AppHeaderProps {
  activeView: AppView;
  onViewChange: (view: AppView) => void;
}

export function AppHeader({ activeView, onViewChange }: AppHeaderProps) {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-canvas/90 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-4 px-4 sm:px-6 lg:px-8">
        <button
          type="button"
          onClick={() => onViewChange("analyze")}
          className="flex shrink-0 items-center gap-2.5"
          aria-label="Open TruthLens analyzer"
        >
          <span className="grid size-9 place-items-center rounded-lg bg-ink text-surface">
            <Scan size={20} weight="bold" />
          </span>
          <span className="text-[17px] font-extrabold tracking-[-0.035em]">TruthLens</span>
        </button>

        <nav className="ml-auto flex items-center gap-1 rounded-xl border border-line bg-surface p-1" aria-label="Main navigation">
          {navigation.map(({ id, label, icon: Icon }) => {
            const active = activeView === id;
            return (
              <button
                key={id}
                type="button"
                onClick={() => onViewChange(id)}
                className={`inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs font-bold transition sm:px-3 ${
                  active ? "bg-surface-subtle text-ink" : "text-muted hover:text-ink"
                }`}
                aria-current={active ? "page" : undefined}
              >
                <Icon size={16} weight={active ? "fill" : "bold"} />
                <span className="hidden sm:inline">{label}</span>
              </button>
            );
          })}
        </nav>

        <div className="hidden items-center gap-2 text-xs font-semibold text-muted lg:flex">
          <span className="size-2 rounded-full bg-warning" aria-hidden="true" />
          Research build
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
}
