import { backendFetch, relayJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return relayJson(await backendFetch("/health"));
  } catch {
    return Response.json({ status: "offline" }, { status: 503 });
  }
}
