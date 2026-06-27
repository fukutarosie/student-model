import OpenAI from "openai";
import { NextResponse } from "next/server";

const SCORE_KEYS = [
  "mouthSmileLeft",
  "mouthSmileRight",
  "browDownLeft",
  "browDownRight",
  "eyeSquintLeft",
  "eyeSquintRight",
  "jawOpen",
  "browInnerUp",
  "mouthFrownLeft",
  "mouthFrownRight",
] as const;

const EXPRESSIONS = [
  "happy",
  "neutral",
  "sad",
  "angry",
  "surprised",
  "tired",
  "unclear",
] as const;

const WARNING =
  "This is only an expression estimate, not a real mood or mental health diagnosis.";

type ScoreKey = (typeof SCORE_KEYS)[number];
type Scores = Record<ScoreKey, number>;
type Expression = (typeof EXPRESSIONS)[number];

type AnalysisResponse = {
  expression: Expression;
  confidence: number;
  reason: string;
  warning: string;
};

export const runtime = "nodejs";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeScores(value: unknown): Scores | null {
  if (!isRecord(value)) {
    return null;
  }

  const scores = {} as Scores;

  for (const key of SCORE_KEYS) {
    const score = value[key];

    if (typeof score !== "number" || !Number.isFinite(score)) {
      return null;
    }

    scores[key] = Math.max(0, Math.min(1, score));
  }

  return scores;
}

function clampConfidence(value: unknown) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(1, value))
    : 0;
}

function normalizeAnalysis(value: unknown): AnalysisResponse {
  if (!isRecord(value)) {
    throw new Error("OpenAI returned an invalid response.");
  }

  const expression = EXPRESSIONS.includes(value.expression as Expression)
    ? (value.expression as Expression)
    : "unclear";
  const reason =
    typeof value.reason === "string"
      ? value.reason
      : "The visible facial expression scores were unclear.";

  return {
    expression,
    confidence: clampConfidence(value.confidence),
    reason,
    warning: WARNING,
  };
}

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
  try {
    let body: unknown;

    try {
      body = await request.json();
    } catch {
      return NextResponse.json(
        { error: "Invalid JSON body." },
        { status: 400 },
      );
    }

    const scores = normalizeScores(isRecord(body) ? body.scores : undefined);

    if (!scores) {
      return NextResponse.json(
        { error: "Missing or invalid scores." },
        { status: 400 },
      );
    }

    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json(
        { error: "OPENAI_API_KEY is not configured." },
        { status: 500 },
      );
    }

    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const completion = await client.chat.completions.create({
      model: "gpt-5.5",
      messages: [
        {
          role: "developer",
          content:
            "Analyze only the visible facial expression based on the provided MediaPipe face blendshape scores. Do not identify the person. Do not infer race, gender, age, health, mental health, or any sensitive attributes. Do not claim to know the user's real mood. Do not diagnose emotional or psychological state. Return strict JSON only with the requested format.",
        },
        {
          role: "user",
          content: JSON.stringify({
            scores,
            allowedExpressions: EXPRESSIONS,
            requiredWarning: WARNING,
          }),
        },
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "visible_expression_analysis",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: {
              expression: {
                type: "string",
                enum: EXPRESSIONS,
              },
              confidence: {
                type: "number",
                minimum: 0,
                maximum: 1,
              },
              reason: {
                type: "string",
                description:
                  "Short reason based only on visible facial expression scores.",
              },
              warning: {
                type: "string",
                enum: [WARNING],
              },
            },
            required: ["expression", "confidence", "reason", "warning"],
          },
        },
      },
    });

    const content = completion.choices[0]?.message.content;

    if (!content) {
      throw new Error("OpenAI returned an empty response.");
    }

    const analysis = normalizeAnalysis(JSON.parse(content));

    return NextResponse.json(analysis);
  } catch (error) {
    console.error("Expression analysis failed:", error);

    return NextResponse.json(
      { error: "Unable to analyze expression." },
      { status: 500 },
    );
  }
}
