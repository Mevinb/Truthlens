import { backendFetch, relayJson } from "@/lib/backend";

export const runtime = "nodejs";
export const maxDuration = 180;

export async function POST(request: Request) {
  const contentLength = Number(request.headers.get("content-length") ?? 0);
  if (contentLength > 27 * 1024 * 1024) {
    return Response.json({ detail: "Images must be 25 MB or smaller." }, { status: 413 });
  }
  try {
    const formData = await request.formData();
    const response = await backendFetch("/analyze", {
      method: "POST",
      body: formData,
    });
    return relayJson(response);
  } catch {
    return Response.json(
      { detail: "The inference service is offline. Start it with ./run.sh." },
      { status: 503 },
    );
  }
}
