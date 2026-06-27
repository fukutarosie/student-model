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

const COMMUNICATION_TONES = [
  "supportive",
  "uncertain",
  "tense",
  "engaged",
  "disengaged",
  "mixed",
] as const;

const WARNING =
  "This combines visible expression and transcript cues only. It is not a diagnosis or proof of a participant's real feelings.";

type ScoreKey = (typeof SCORE_KEYS)[number];
type Scores = Record<ScoreKey, number>;
type Expression = (typeof EXPRESSIONS)[number];
type CommunicationTone = (typeof COMMUNICATION_TONES)[number];
type ScoreSample = {
  timestamp: number;
  scores: Scores;
};
type ExpressionResult = {
  expression: Expression;
  confidence: number;
  reason?: string;
};

type CommunicationResponse = {
  summary: string;
  communicationTone: CommunicationTone;
  confidence: number;
  visibleExpression: Expression;
  audioContentExpression: Expression;
  finalExpression: Expression;
  audioContentScores: Record<Expression, number>;
  facialExpressionScores: Record<Expression, number>;
  combinedExpressionScores: Record<Expression, number>;
  spokenSignals: string[];
  facialSignals: string[];
  recommendation: string;
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

function normalizeSamples(value: unknown, fallbackScores: Scores) {
  if (!Array.isArray(value)) {
    return [{ timestamp: 0, scores: fallbackScores }];
  }

  const samples: ScoreSample[] = [];

  for (const item of value.slice(-64)) {
    if (!isRecord(item) || typeof item.timestamp !== "number") {
      continue;
    }

    const scores = normalizeScores(item.scores);

    if (scores) {
      samples.push({ timestamp: item.timestamp, scores });
    }
  }

  return samples.length > 0 ? samples : [{ timestamp: 0, scores: fallbackScores }];
}

function normalizeTranscript(value: unknown) {
  if (typeof value !== "string") {
    return "";
  }

  return value.replace(/\s+/g, " ").trim().slice(0, 6000);
}

function average(left: number, right: number) {
  return (left + right) / 2;
}

function round(value: number) {
  return Math.round(value * 1000) / 1000;
}

function getMean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function getExpressionResult(value: unknown): ExpressionResult | null {
  if (!isRecord(value) || !EXPRESSIONS.includes(value.expression as Expression)) {
    return null;
  }

  return {
    expression: value.expression as Expression,
    confidence:
      typeof value.confidence === "number" && Number.isFinite(value.confidence)
        ? Math.max(0, Math.min(1, value.confidence))
        : 0,
    reason: typeof value.reason === "string" ? value.reason : undefined,
  };
}

function getFacialSummary(samples: ScoreSample[], latestScores: Scores) {
  const smileSeries = samples.map((sample) =>
    average(sample.scores.mouthSmileLeft, sample.scores.mouthSmileRight),
  );
  const browSeries = samples.map((sample) =>
    average(sample.scores.browDownLeft, sample.scores.browDownRight),
  );
  const eyeSeries = samples.map((sample) =>
    average(sample.scores.eyeSquintLeft, sample.scores.eyeSquintRight),
  );
  const frownSeries = samples.map((sample) =>
    average(sample.scores.mouthFrownLeft, sample.scores.mouthFrownRight),
  );

  return {
    latest: {
      smile: round(average(latestScores.mouthSmileLeft, latestScores.mouthSmileRight)),
      browTension: round(average(latestScores.browDownLeft, latestScores.browDownRight)),
      eyeNarrowing: round(average(latestScores.eyeSquintLeft, latestScores.eyeSquintRight)),
      frown: round(average(latestScores.mouthFrownLeft, latestScores.mouthFrownRight)),
      jawOpen: round(latestScores.jawOpen),
      innerBrowUp: round(latestScores.browInnerUp),
    },
    window: {
      smileMean: round(getMean(smileSeries)),
      browTensionMean: round(getMean(browSeries)),
      eyeNarrowingMean: round(getMean(eyeSeries)),
      frownMean: round(getMean(frownSeries)),
      sampleCount: samples.length,
    },
  };
}

function normalizeRange(value: number, low: number, high: number) {
  if (high <= low) {
    return 0;
  }

  return Math.max(0, Math.min(1, (value - low) / (high - low)));
}

function getFacialExpressionScores(facialSummary: ReturnType<typeof getFacialSummary>) {
  const smile = normalizeRange(facialSummary.window.smileMean, 0.12, 0.5);
  const brow = normalizeRange(facialSummary.window.browTensionMean, 0.12, 0.44);
  const squint = normalizeRange(facialSummary.window.eyeNarrowingMean, 0.1, 0.4);
  const frown = normalizeRange(facialSummary.window.frownMean, 0.1, 0.42);
  const jaw = normalizeRange(facialSummary.latest.jawOpen, 0.14, 0.55);
  const innerBrow = normalizeRange(facialSummary.latest.innerBrowUp, 0.12, 0.45);
  const activation = Math.max(smile, brow, squint, frown, jaw, innerBrow);
  const lowSmile = 1 - smile;

  return {
    happy: round(Math.max(0, Math.min(1, 0.74 * smile + 0.1 * squint - 0.18 * frown - 0.14 * brow))),
    neutral: round(Math.max(0, Math.min(1, 1 - normalizeRange(activation, 0.08, 0.35)))),
    sad: round(Math.max(0, Math.min(1, 0.42 * frown + 0.34 * innerBrow + 0.14 * lowSmile - 0.1 * smile))),
    angry: round(Math.max(0, Math.min(1, 0.42 * brow + 0.28 * squint + 0.18 * frown + 0.08 * lowSmile - 0.12 * smile))),
    surprised: round(Math.max(0, Math.min(1, 0.46 * jaw + 0.34 * innerBrow + 0.1 * lowSmile - 0.12 * squint))),
    tired: round(Math.max(0, Math.min(1, 0.42 * squint + 0.18 * innerBrow + 0.16 * lowSmile - 0.14 * jaw))),
    unclear: 0,
  } satisfies Record<Expression, number>;
}
function getOpenAIErrorMessage(error: unknown) {
  if (!isRecord(error)) {
    return "Unable to analyze communication.";
  }

  const status = typeof error.status === "number" ? error.status : undefined;
  const code = typeof error.code === "string" ? error.code : undefined;
  const message = typeof error.message === "string" ? error.message : "";

  if (status === 401) {
    return "OpenAI API key was rejected. Check OPENAI_API_KEY on the server.";
  }

  if (status === 429) {
    return "OpenAI rate limit or quota was reached. Try again later or check billing.";
  }

  if (code === "ENOTFOUND" || message.toLowerCase().includes("connection error")) {
    return "Cannot reach OpenAI API. Check network access, proxy, VPN, or DNS.";
  }

  return "Unable to analyze communication.";
}

function normalizeCommunicationResponse(value: unknown): CommunicationResponse {
  if (!isRecord(value)) {
    throw new Error("OpenAI returned an invalid communication response.");
  }

  const normalizeExpressionScores = (scores: unknown) => {
    const output = Object.fromEntries(
      EXPRESSIONS.map((expression) => [expression, 0]),
    ) as Record<Expression, number>;

    if (!isRecord(scores)) {
      return output;
    }

    for (const expression of EXPRESSIONS) {
      const score = scores[expression];

      if (typeof score === "number" && Number.isFinite(score)) {
        output[expression] = Math.max(0, Math.min(1, score));
      }
    }

    return output;
  };

  return {
    summary:
      typeof value.summary === "string"
        ? value.summary
        : "The communication signals were unclear.",
    communicationTone: COMMUNICATION_TONES.includes(
      value.communicationTone as CommunicationTone,
    )
      ? (value.communicationTone as CommunicationTone)
      : "mixed",
    confidence:
      typeof value.confidence === "number" && Number.isFinite(value.confidence)
        ? Math.max(0, Math.min(1, value.confidence))
        : 0,
    visibleExpression: EXPRESSIONS.includes(value.visibleExpression as Expression)
      ? (value.visibleExpression as Expression)
      : "unclear",
    audioContentExpression: EXPRESSIONS.includes(
      value.audioContentExpression as Expression,
    )
      ? (value.audioContentExpression as Expression)
      : "unclear",
    finalExpression: EXPRESSIONS.includes(value.finalExpression as Expression)
      ? (value.finalExpression as Expression)
      : "unclear",
    audioContentScores: normalizeExpressionScores(value.audioContentScores),
    facialExpressionScores: normalizeExpressionScores(value.facialExpressionScores),
    combinedExpressionScores: normalizeExpressionScores(
      value.combinedExpressionScores,
    ),
    spokenSignals: Array.isArray(value.spokenSignals)
      ? value.spokenSignals
          .filter((signal): signal is string => typeof signal === "string")
          .slice(0, 4)
      : [],
    facialSignals: Array.isArray(value.facialSignals)
      ? value.facialSignals
          .filter((signal): signal is string => typeof signal === "string")
          .slice(0, 4)
      : [],
    recommendation:
      typeof value.recommendation === "string"
        ? value.recommendation
        : "Ask a short clarifying question and give the participant time to answer.",
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
        { error: "Missing or invalid facial expression scores." },
        { status: 400 },
      );
    }

    const transcript = normalizeTranscript(
      isRecord(body) ? body.transcript : undefined,
    );

    if (!transcript) {
      return NextResponse.json(
        { error: "Missing transcript. Start voice listening or recording first." },
        { status: 400 },
      );
    }

    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json(
        { error: "OPENAI_API_KEY is not configured." },
        { status: 500 },
      );
    }

    const samples = normalizeSamples(isRecord(body) ? body.samples : undefined, scores);
    const expressionResult = getExpressionResult(
      isRecord(body) ? body.expressionResult : undefined,
    );
    const facialSummary = getFacialSummary(samples, scores);
    const facialExpressionScores = getFacialExpressionScores(facialSummary);
    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const completion = await client.chat.completions.create({
      model: process.env.OPENAI_MODEL ?? "gpt-5.5",
      messages: [
        {
          role: "developer",
          content:
            "Analyze classroom or meeting communication using only the participant transcript and visible facial-expression signal summary. Score the transcript/audio content for expressed affect cues across the allowed expressions, then combine those audio-content scores with the provided facial-expression scores to produce a final combined expression. Treat transcript/audio-content cues as 60% of the final score and visible facial cues as 40%, unless transcript content is too short or ambiguous; in that case reduce confidence and prefer unclear/neutral. Do not identify the person. Do not infer protected or sensitive traits. Do not claim to know real emotions, intent, truthfulness, mental health, or personality. Use wording such as expressed by the words and visible cues suggest rather than they feel. Return practical, respectful coaching that helps the listener communicate better.",
        },
        {
          role: "user",
          content: JSON.stringify({
            transcript,
            expressionResult,
            facialSummary,
            facialExpressionScores,
            scoreFusionGuidance: "combinedExpressionScores should roughly follow audioContentScores * 0.6 + facialExpressionScores * 0.4, with lower confidence if the transcript and face cues conflict.",
            allowedTones: COMMUNICATION_TONES,
            allowedExpressions: EXPRESSIONS,
            requiredWarning: WARNING,
          }),
        },
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "communication_analysis",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: {
              summary: {
                type: "string",
                description:
                  "One concise paragraph about the communication cues.",
              },
              communicationTone: {
                type: "string",
                enum: COMMUNICATION_TONES,
              },
              confidence: {
                type: "number",
                minimum: 0,
                maximum: 1,
              },
              visibleExpression: {
                type: "string",
                enum: EXPRESSIONS,
              },
              audioContentExpression: {
                type: "string",
                enum: EXPRESSIONS,
                description:
                  "Best expression label from transcript/audio-content cues only.",
              },
              finalExpression: {
                type: "string",
                enum: EXPRESSIONS,
                description:
                  "Best final expression label after combining transcript/audio content and facial cues.",
              },
              audioContentScores: {
                type: "object",
                additionalProperties: false,
                properties: Object.fromEntries(
                  EXPRESSIONS.map((expression) => [
                    expression,
                    { type: "number", minimum: 0, maximum: 1 },
                  ]),
                ),
                required: EXPRESSIONS,
              },
              facialExpressionScores: {
                type: "object",
                additionalProperties: false,
                properties: Object.fromEntries(
                  EXPRESSIONS.map((expression) => [
                    expression,
                    { type: "number", minimum: 0, maximum: 1 },
                  ]),
                ),
                required: EXPRESSIONS,
              },
              combinedExpressionScores: {
                type: "object",
                additionalProperties: false,
                properties: Object.fromEntries(
                  EXPRESSIONS.map((expression) => [
                    expression,
                    { type: "number", minimum: 0, maximum: 1 },
                  ]),
                ),
                required: EXPRESSIONS,
              },
              spokenSignals: {
                type: "array",
                items: { type: "string" },
                minItems: 1,
                maxItems: 4,
              },
              facialSignals: {
                type: "array",
                items: { type: "string" },
                minItems: 1,
                maxItems: 4,
              },
              recommendation: {
                type: "string",
                description:
                  "A respectful next-step recommendation for communicating better.",
              },
              warning: {
                type: "string",
                enum: [WARNING],
              },
            },
            required: [
              "summary",
              "communicationTone",
              "confidence",
              "visibleExpression",
              "audioContentExpression",
              "finalExpression",
              "audioContentScores",
              "facialExpressionScores",
              "combinedExpressionScores",
              "spokenSignals",
              "facialSignals",
              "recommendation",
              "warning",
            ],
          },
        },
      },
    });

    const content = completion.choices[0]?.message.content;

    if (!content) {
      throw new Error("OpenAI returned an empty response.");
    }

    return NextResponse.json(
      normalizeCommunicationResponse(JSON.parse(content)),
    );
  } catch (error: unknown) {
    console.error("Communication analysis failed:", error);

    return NextResponse.json(
      { error: getOpenAIErrorMessage(error) },
      { status: 502 },
    );
  }
}









