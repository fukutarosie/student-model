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
  "focused",
  "serious",
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
  "This combines transcript cues and visible expression estimates only. It is not proof of a real mood, intent, or mental health state.";

type ScoreKey = (typeof SCORE_KEYS)[number];
type Scores = Record<ScoreKey, number>;
type NumericScores = Record<string, number>;
type Expression = (typeof EXPRESSIONS)[number];
type CommunicationTone = (typeof COMMUNICATION_TONES)[number];

type ScoreSample = {
  timestamp: number;
  scores: Scores;
  allScores: NumericScores;
  visionMetrics: unknown;
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

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function roundScore(value: number) {
  return Math.round(value * 1000) / 1000;
}

function average(left: number, right: number) {
  return (left + right) / 2;
}

function normalizeRange(value: number, low: number, high: number) {
  if (high <= low) {
    return 0;
  }

  return clamp01((value - low) / (high - low));
}

function getMean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
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

    scores[key] = clamp01(score);
  }

  return scores;
}

function normalizeNumericScores(value: unknown, fallback: NumericScores = {}) {
  if (!isRecord(value)) {
    return fallback;
  }

  const scores: NumericScores = {};

  for (const [key, score] of Object.entries(value)) {
    if (typeof score === "number" && Number.isFinite(score)) {
      scores[key] = clamp01(score);
    }
  }

  return Object.keys(scores).length > 0 ? scores : fallback;
}

function sanitizeJsonValue(value: unknown, depth = 0): unknown {
  if (depth > 4) {
    return undefined;
  }

  if (
    typeof value === "string" ||
    typeof value === "boolean" ||
    value === null
  ) {
    return value;
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }

  if (Array.isArray(value)) {
    return value
      .map((item) => sanitizeJsonValue(item, depth + 1))
      .filter((item) => item !== undefined)
      .slice(0, 80);
  }

  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .map(([key, item]) => [key, sanitizeJsonValue(item, depth + 1)])
        .filter(([, item]) => item !== undefined),
    );
  }

  return undefined;
}

function normalizeSamples(
  value: unknown,
  fallbackScores: Scores,
  fallbackAllScores: NumericScores,
) {
  if (!Array.isArray(value)) {
    return [
      {
        timestamp: 0,
        scores: fallbackScores,
        allScores: fallbackAllScores,
        visionMetrics: null,
      },
    ];
  }

  const samples: ScoreSample[] = [];

  for (const item of value.slice(-64)) {
    if (!isRecord(item) || typeof item.timestamp !== "number") {
      continue;
    }

    const scores = normalizeScores(item.scores);

    if (!scores) {
      continue;
    }

    samples.push({
      timestamp: item.timestamp,
      scores,
      allScores: normalizeNumericScores(item.allScores, fallbackAllScores),
      visionMetrics: sanitizeJsonValue(item.visionMetrics) ?? null,
    });
  }

  return samples.length > 0
    ? samples
    : [
        {
          timestamp: 0,
          scores: fallbackScores,
          allScores: fallbackAllScores,
          visionMetrics: null,
        },
      ];
}

function normalizeTranscript(value: unknown) {
  if (typeof value !== "string") {
    return "";
  }

  return value.replace(/\s+/g, " ").trim().slice(0, 6000);
}

function getExpression(value: unknown): Expression {
  return EXPRESSIONS.includes(value as Expression)
    ? (value as Expression)
    : "unclear";
}

function getExpressionResult(value: unknown): ExpressionResult | null {
  if (!isRecord(value) || !EXPRESSIONS.includes(value.expression as Expression)) {
    return null;
  }

  return {
    expression: value.expression as Expression,
    confidence:
      typeof value.confidence === "number" && Number.isFinite(value.confidence)
        ? clamp01(value.confidence)
        : 0,
    reason: typeof value.reason === "string" ? value.reason : undefined,
  };
}

function getFacialSummary(params: {
  samples: ScoreSample[];
  latestScores: Scores;
  allScores: NumericScores;
  visionMetrics: unknown;
}) {
  const { samples, latestScores, allScores, visionMetrics } = params;
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
      smile: roundScore(
        average(latestScores.mouthSmileLeft, latestScores.mouthSmileRight),
      ),
      browTension: roundScore(
        average(latestScores.browDownLeft, latestScores.browDownRight),
      ),
      eyeNarrowing: roundScore(
        average(latestScores.eyeSquintLeft, latestScores.eyeSquintRight),
      ),
      frown: roundScore(
        average(latestScores.mouthFrownLeft, latestScores.mouthFrownRight),
      ),
      jawOpen: roundScore(latestScores.jawOpen),
      innerBrowUp: roundScore(latestScores.browInnerUp),
    },
    window: {
      smileMean: roundScore(getMean(smileSeries)),
      browTensionMean: roundScore(getMean(browSeries)),
      eyeNarrowingMean: roundScore(getMean(eyeSeries)),
      frownMean: roundScore(getMean(frownSeries)),
      sampleCount: samples.length,
    },
    allBlendshapeScoreCount: Object.keys(allScores).length,
    visionMetrics: sanitizeJsonValue(visionMetrics),
  };
}

function getFacialExpressionScores(
  facialSummary: ReturnType<typeof getFacialSummary>,
) {
  const smile = normalizeRange(facialSummary.window.smileMean, 0.16, 0.52);
  const brow = normalizeRange(facialSummary.window.browTensionMean, 0.12, 0.44);
  const squint = normalizeRange(
    facialSummary.window.eyeNarrowingMean,
    0.1,
    0.4,
  );
  const frown = normalizeRange(facialSummary.window.frownMean, 0.1, 0.42);
  const jaw = normalizeRange(facialSummary.latest.jawOpen, 0.14, 0.55);
  const innerBrow = normalizeRange(facialSummary.latest.innerBrowUp, 0.12, 0.45);
  const activation = Math.max(smile, brow, squint, frown, jaw, innerBrow);
  const lowSmile = 1 - smile;
  const lowJaw = 1 - jaw;
  const browEyeAgreement = Math.min(brow, squint);

  return {
    happy: roundScore(clamp01(0.72 * smile + 0.12 * squint - 0.22 * frown - 0.18 * brow)),
    neutral: roundScore(clamp01(1 - normalizeRange(activation, 0.08, 0.32))),
    focused: roundScore(
      clamp01(0.34 * squint + 0.24 * lowSmile + 0.18 * lowJaw + 0.16 * activation - 0.16 * frown - 0.14 * innerBrow),
    ),
    serious: roundScore(
      clamp01(0.28 * lowSmile + 0.22 * frown + 0.2 * brow + 0.14 * lowJaw - 0.18 * smile - 0.12 * jaw),
    ),
    sad: roundScore(
      clamp01(0.42 * frown + 0.35 * innerBrow + 0.16 * lowSmile - 0.12 * brow - 0.08 * smile),
    ),
    angry: roundScore(
      clamp01(0.36 * browEyeAgreement + 0.22 * brow + 0.18 * squint + 0.2 * frown + 0.04 * lowSmile - 0.16 * smile - 0.1 * innerBrow - 0.1 * jaw),
    ),
    surprised: roundScore(
      clamp01(0.46 * jaw + 0.34 * innerBrow + 0.12 * lowSmile - 0.16 * squint - 0.12 * brow),
    ),
    tired: roundScore(
      clamp01(0.42 * squint + 0.18 * innerBrow + 0.16 * lowSmile - 0.18 * brow - 0.12 * jaw),
    ),
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
  const cause = isRecord(error.cause) ? error.cause : undefined;
  const causeCode = typeof cause?.code === "string" ? cause.code : undefined;
  const causeHostname =
    typeof cause?.hostname === "string" ? cause.hostname : undefined;

  if (status === 401) {
    return "OpenAI API key was rejected. Check OPENAI_API_KEY on the server.";
  }

  if (status === 429) {
    return "OpenAI rate limit or quota was reached. Try again later or check billing.";
  }

  if (
    code === "ENOTFOUND" ||
    causeCode === "ENOTFOUND" ||
    message.toLowerCase().includes("connection error") ||
    message.toLowerCase().includes("timed out")
  ) {
    return `Cannot reach OpenAI API${causeHostname ? ` (${causeHostname})` : ""}. Check DNS, proxy, VPN, or network access, then try again.`;
  }

  return "Unable to analyze communication.";
}

function normalizeExpressionScores(value: unknown) {
  const output = Object.fromEntries(
    EXPRESSIONS.map((expression) => [expression, 0]),
  ) as Record<Expression, number>;

  if (!isRecord(value)) {
    return output;
  }

  for (const expression of EXPRESSIONS) {
    const score = value[expression];

    if (typeof score === "number" && Number.isFinite(score)) {
      output[expression] = clamp01(score);
    }
  }

  return output;
}

function getStringArray(value: unknown, fallback: string[]) {
  if (!Array.isArray(value)) {
    return fallback;
  }

  const strings = value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 4);

  return strings.length > 0 ? strings : fallback;
}

function normalizeCommunicationResponse(value: unknown): CommunicationResponse {
  if (!isRecord(value)) {
    throw new Error("OpenAI returned an invalid communication response.");
  }

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
        ? clamp01(value.confidence)
        : 0,
    visibleExpression: getExpression(value.visibleExpression),
    audioContentExpression: getExpression(value.audioContentExpression),
    finalExpression: getExpression(value.finalExpression),
    audioContentScores: normalizeExpressionScores(value.audioContentScores),
    facialExpressionScores: normalizeExpressionScores(value.facialExpressionScores),
    combinedExpressionScores: normalizeExpressionScores(
      value.combinedExpressionScores,
    ),
    spokenSignals: getStringArray(value.spokenSignals, [
      "The transcript did not provide a strong spoken signal.",
    ]),
    facialSignals: getStringArray(value.facialSignals, [
      "Visible expression signals were weak or mixed.",
    ]),
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
        { error: "Missing transcript. Start voice listening or type words first." },
        { status: 400 },
      );
    }

    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json(
        { error: "OPENAI_API_KEY is not configured." },
        { status: 500 },
      );
    }

    const allScores = normalizeNumericScores(
      isRecord(body) ? body.allScores : undefined,
      scores,
    );
    const visionMetrics =
      sanitizeJsonValue(isRecord(body) ? body.visionMetrics : undefined) ??
      null;
    const samples = normalizeSamples(
      isRecord(body) ? body.samples : undefined,
      scores,
      allScores,
    );
    const expressionResult = getExpressionResult(
      isRecord(body) ? body.expressionResult : undefined,
    );
    const facialSummary = getFacialSummary({
      samples,
      latestScores: scores,
      allScores,
      visionMetrics,
    });
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
            "Analyze communication using only the speech transcript text and browser-local numeric visible facial-expression signals. The transcript may come from browser speech recognition and may be imperfect. Do not use or imply access to audio, video, images, identity, hidden intent, real mood, truthfulness, personality, health, mental health, race, gender, age, or other sensitive traits. Score transcript wording for expressed communication cues, score visible expression cues from the numeric data, and combine them into a cautious communication interpretation. Use wording like the words suggest and visible cues suggest. Treat transcript cues as roughly 60% and visible expression cues as roughly 40%, unless the transcript is too short, ambiguous, or conflicts with the face cues. Return practical, respectful communication coaching. Return strict JSON only in English.",
        },
        {
          role: "user",
          content: JSON.stringify({
            transcript,
            expressionResult,
            facialSummary,
            facialExpressionScores,
            scoreFusionGuidance:
              "combinedExpressionScores should roughly follow transcript/audio-content scores * 0.6 plus facialExpressionScores * 0.4, with lower confidence if the transcript and face cues conflict.",
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
                  "One concise paragraph about transcript and visible-expression communication cues.",
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
                  "Best expression label from transcript wording only.",
              },
              finalExpression: {
                type: "string",
                enum: EXPRESSIONS,
                description:
                  "Best final label after combining transcript and visible cues.",
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
