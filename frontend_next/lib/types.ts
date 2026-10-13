export type AppView = "analyze" | "benchmarks" | "method";
export type QueueStatus = "ready" | "analyzing" | "complete" | "error";

export interface ModelOption {
  key: string;
  label: string;
  shortLabel: string;
  description: string;
  group: "Recommended" | "Deep learning" | "Classical";
  supportsGradcam?: boolean;
}

export interface PredictionResult {
  prediction: "REAL" | "FAKE";
  confidence: number;
  probabilities: { REAL: number; FAKE: number };
  review_recommended?: boolean;
  reliability?: string;
  explanation?: string[];
  likely_generator?: string;
  model_name?: string;
  decision_threshold?: number;
  raw_probability_fake?: number;
  view_probabilities?: Record<string, number>;
  view_disagreement?: number;
  spectral_analysis?: Record<string, unknown>;
  ela_metrics?: Record<string, unknown>;
  provenance_analysis?: Record<string, unknown>;
  ensemble_votes?: Record<string, unknown> | unknown[];
  [key: string]: unknown;
}

export interface AnalysisResponse {
  filename: string;
  width: number;
  height: number;
  resolution_tier: string;
  model_type: string;
  result: PredictionResult;
  heatmap: string | null;
}

export interface QueueItem {
  id: string;
  file: File;
  preview: string;
  status: QueueStatus;
  response?: AnalysisResponse;
  error?: string;
}

export interface BenchmarkRow {
  Model: string;
  Accuracy: number;
  F1: number;
  AUC: number;
  "Real acc"?: number;
  "Fake acc"?: number;
  Images?: number;
}

export interface ResolutionRow {
  "Resolution tier": string;
  Accuracy: number;
  F1: number;
  AUC: number;
  "Real acc"?: number;
  "Fake acc"?: number;
  Images?: number;
}
