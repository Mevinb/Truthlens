"use client";

import { useState } from "react";
import { Analyzer } from "@/components/analyzer";
import { AppHeader } from "@/components/app-header";
import { Benchmarks } from "@/components/benchmarks";
import { Method } from "@/components/method";
import type { AppView } from "@/lib/types";

export function AppShell() {
  const [activeView, setActiveView] = useState<AppView>("analyze");

  return (
    <div className="min-h-screen">
      <AppHeader activeView={activeView} onViewChange={setActiveView} />
      <main>
        {activeView === "analyze" && <Analyzer />}
        {activeView === "benchmarks" && <Benchmarks />}
        {activeView === "method" && <Method />}
      </main>
      <footer className="border-t border-line py-7">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 text-xs text-muted sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
          <p>TruthLens is a research tool. Treat every result as evidence, not proof.</p>
          <p>Local model pipeline · Training in progress</p>
        </div>
      </footer>
    </div>
  );
}
