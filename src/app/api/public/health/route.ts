import { NextResponse } from "next/server";

export const runtime = "nodejs";

export function GET() {
  return NextResponse.json({
    name: "Facial Communication Recognition API",
    status: "ok",
    version: "1.0",
    endpoints: {
      expression: "POST /api/public/expression",
      communication: "POST /api/public/communication",
    },
    authentication: "Send x-api-key or Authorization: Bearer <key>.",
  });
}

