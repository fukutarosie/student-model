import { NextResponse } from "next/server";
import { validatePublicApiKey } from "../_auth";

export const runtime = "nodejs";

export function GET() {
  return NextResponse.json(
    { error: "Method not allowed. Use POST." },
    {
      status: 405,
      headers: {
        Allow: "POST",
      },
    },
  );
}

export async function POST(request: Request) {
  const authError = validatePublicApiKey(request);

  if (authError) {
    return authError;
  }

  const target = new URL("/api/analyze-expression", request.url);
  const response = await fetch(target, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: await request.text(),
  });

  return NextResponse.json(await response.json(), { status: response.status });
}

