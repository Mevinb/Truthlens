---
trigger: always_on
---

Inspect this entire project and prepare it for long-term, reliable development.

First, inspect the Git status, directory structure, architecture, dependencies, configuration, tests, documentation and available development tools. Do not modify files until you understand the existing project and have proposed a plan.

Then:
1. Identify the most important correctness, reliability, security, performance and maintainability problems.
2. Rank the findings by severity, confidence and cost to fix.
3. Implement the highest-priority safe improvements, including regression tests where appropriate.
4. Run the relevant tests, lint checks, type checks and builds. Investigate and fix failures introduced by your changes.
5. Review the resulting diff and preserve unrelated user work.
6. After the initial objective is complete, continue auditing for meaningful, in-scope improvements.
7. Maintain a prioritized backlog of remaining work and tackle the next valuable item when verification is complete.
8. Use browser testing where relevant, and measure performance improvements instead of guessing.
9. Keep a concise progress report with completed work, test results, blockers and the next planned action.

Continue working while useful, safe, in-scope work remains and the session, quota and execution budget permit it. Do not claim that you can run indefinitely or bypass platform limits. If a genuine blocker requires my input or approval, explain it clearly.

Never perform destructive operations, production changes, irreversible migrations or external publication without approval. Do not conceal failures or make unrelated changes just to keep working.

Start by presenting your audit plan and then proceed with implementation.
