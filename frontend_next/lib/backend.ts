const BACKEND_URL = process.env.TRUTHLENS_API_URL ?? "http://127.0.0.1:8000";

export async function backendFetch(path: string, init?: RequestInit) {
  return fetch(`${BACKEND_URL}${path}`, {
    ...init,
    cache: "no-store",
  });
}

export async function relayJson(response: Response) {
  const body = await response.text();
  return new Response(body, {
    status: response.status,
    headers: { "content-type": "application/json" },
  });
}
