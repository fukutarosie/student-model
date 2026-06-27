import { NextResponse } from "next/server";

function getConfiguredKeys() {
  const combinedKeys = [
    process.env.FACIAL_PUBLIC_API_KEY,
    process.env.FACIAL_API_KEYS,
  ]
    .filter(Boolean)
    .join(",");

  return combinedKeys
    .split(",")
    .map((key) => key.trim())
    .filter(Boolean);
}

function getRequestKey(request: Request) {
  const bearer = request.headers.get("authorization")?.match(/^Bearer\s+(.+)$/i);

  return request.headers.get("x-api-key") ?? bearer?.[1] ?? "";
}

export function validatePublicApiKey(request: Request) {
  const configuredKeys = getConfiguredKeys();

  if (configuredKeys.length === 0) {
    return NextResponse.json(
      {
        error:
          "Public API access is not configured. Set FACIAL_PUBLIC_API_KEY or FACIAL_API_KEYS.",
      },
      { status: 500 },
    );
  }

  const requestKey = getRequestKey(request);

  if (!requestKey || !configuredKeys.includes(requestKey)) {
    return NextResponse.json(
      { error: "Invalid or missing API key." },
      { status: 401 },
    );
  }

  return null;
}

