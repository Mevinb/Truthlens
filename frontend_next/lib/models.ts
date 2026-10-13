import type { ModelOption } from "@/lib/types";

export const MODELS: ModelOption[] = [
  {
    key: "swin_improved_ep6",
    label: "SwinV2 Tiny, improved epoch 6",
    shortLabel: "SwinV2 Ep 6",
    description: "Best overall Swin checkpoint across the current project benchmarks.",
    group: "Recommended",
  },
  {
    key: "clip",
    label: "CLIP ViT-L/14, retrained",
    shortLabel: "CLIP ViT-L/14",
    description: "Strong test split accuracy with calibrated multi-view linear heads.",
    group: "Recommended",
  },
  {
    key: "swin_improved_ep5",
    label: "SwinV2 Tiny, improved epoch 5",
    shortLabel: "SwinV2 Ep 5",
    description: "Lowest validation loss in the improved training run.",
    group: "Deep learning",
  },
  {
    key: "swin_improved_ep3",
    label: "SwinV2 Tiny, improved epoch 3",
    shortLabel: "SwinV2 Ep 3",
    description: "Strong independent web-photo generalization in current tests.",
    group: "Deep learning",
  },
  {
    key: "swin_improved_ep4",
    label: "SwinV2 Tiny, improved epoch 4",
    shortLabel: "SwinV2 Ep 4",
    description: "Aggressive checkpoint optimized for detecting generated images.",
    group: "Deep learning",
  },
  {
    key: "swin_hardstyles",
    label: "SwinV2 Tiny, hard styles",
    shortLabel: "SwinV2 Hard Styles",
    description: "Fine-tuned on difficult generator styles and modern outputs.",
    group: "Deep learning",
  },
  {
    key: "swin_ep9",
    label: "SwinV2 Tiny, epoch 9",
    shortLabel: "SwinV2 Ep 9",
    description: "Union checkpoint with strong results on uncurated web photos.",
    group: "Deep learning",
  },
  {
    key: "resnet18",
    label: "ResNet18 with test-time augmentation",
    shortLabel: "ResNet18",
    description: "CNN baseline with optional Grad-CAM attention visualization.",
    group: "Deep learning",
    supportsGradcam: true,
  },
  {
    key: "ensemble",
    label: "ResNet18 with advisory signals",
    shortLabel: "ResNet18 + signals",
    description: "ResNet18 verdict with classical and forensic signals shown separately.",
    group: "Deep learning",
    supportsGradcam: true,
  },
  {
    key: "logistic_regression",
    label: "Logistic regression",
    shortLabel: "Logistic regression",
    description: "Linear classifier using PCA-reduced HOG and LBP features.",
    group: "Classical",
  },
  {
    key: "random_forest",
    label: "Random forest",
    shortLabel: "Random forest",
    description: "Tree ensemble trained on HOG and LBP texture features.",
    group: "Classical",
  },
  {
    key: "svm",
    label: "SVM with RBF kernel",
    shortLabel: "SVM",
    description: "Nonlinear classifier operating on PCA-reduced visual features.",
    group: "Classical",
  },
  {
    key: "knn",
    label: "k-nearest neighbours",
    shortLabel: "k-NN",
    description: "Neighbour-based prediction in the handcrafted feature space.",
    group: "Classical",
  },
  {
    key: "decision_tree",
    label: "Decision tree",
    shortLabel: "Decision tree",
    description: "Single depth-limited tree on texture and edge features.",
    group: "Classical",
  },
  {
    key: "naive_bayes",
    label: "Naive Bayes",
    shortLabel: "Naive Bayes",
    description: "Gaussian baseline using raw handcrafted features.",
    group: "Classical",
  },
];

export const MODEL_GROUPS = ["Recommended", "Deep learning", "Classical"] as const;

export const SAMPLES = [
  { src: "/samples/real-portrait.jpg", name: "Natural portrait", truth: "REAL" },
  { src: "/samples/real-red-scarf.jpg", name: "Studio portrait", truth: "REAL" },
  { src: "/samples/ai-bird.jpg", name: "Generated bird", truth: "FAKE" },
  { src: "/samples/ai-autumn.jpg", name: "Generated autumn", truth: "FAKE" },
] as const;
