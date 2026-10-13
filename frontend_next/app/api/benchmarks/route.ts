import { backendFetch, relayJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return relayJson(await backendFetch("/benchmarks"));
  } catch {
    return Response.json({ models: [], resolutions: [] }, { status: 503 });
  }
}
