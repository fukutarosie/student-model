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
type ScoreSample = {
  timestamp: number;
  scores: Scores;
};

type ScoreStats = {
  latest: number;
  mean: number;
  max: number;
  min: number;
  delta: number;
  volatility: number;
  stability: number;
};

type TemporalFeatures = {
  sampleCount: number;
  windowMs: number;
  scoreStats: Record<ScoreKey, ScoreStats>;
  smartScores: Scores;
  aggregate: {
    angryTensionMean: number;
    angryTensionPeak: number;
    angryTensionTrend: number;
    angryTensionStableRatio: number;
    smileMean: number;
    smilePeak: number;
    browTensionMean: number;
    browTensionPeak: number;
    eyeNarrowingMean: number;
    frownMean: number;
  };
};

type AnalysisResponse = {
  expression: Expression;
  confidence: number;
  reason: string;
  warning: string;
};

type DerivedFeatures = {
  smileAverage: number;
  frownAverage: number;
  browTension: number;
  eyeNarrowing: number;
  jawOpening: number;
  innerBrowLift: number;
  smileFrownBalance: number;
  leftRightSmileAsymmetry: number;
  activationLevel: number;
  temporalFeatures: TemporalFeatures;
  strongestScores: Array<{ name: ScoreKey; score: number }>;
  expressionScores: Record<Expression, number>;
  heuristicClassification: {
    expression: Expression;
    confidence: number;
    evidence: string[];
  };
  heuristicHints: string[];
};

const MAX_SCORE_SAMPLES = 64;
const MAX_SAMPLE_WINDOW_MS = 6000;

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

function normalizeScoreSamples(value: unknown, fallbackScores: Scores) {
  if (!Array.isArray(value)) {
    return [{ timestamp: 0, scores: fallbackScores }];
  }

  const samples: ScoreSample[] = [];

  for (const item of value) {
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
    });
  }

  if (samples.length === 0) {
    return [{ timestamp: 0, scores: fallbackScores }];
  }

  const sortedSamples = samples
    .sort((left, right) => left.timestamp - right.timestamp)
    .slice(-MAX_SCORE_SAMPLES);
  const latestTimestamp =
    sortedSamples[sortedSamples.length - 1]?.timestamp ?? sortedSamples[0].timestamp;
  const windowedSamples = sortedSamples.filter(
    (sample) => latestTimestamp - sample.timestamp <= MAX_SAMPLE_WINDOW_MS,
  );

  return windowedSamples.length > 0
    ? windowedSamples
    : [{ timestamp: 0, scores: fallbackScores }];
}

function clampConfidence(value: unknown) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(1, value))
    : 0;
}

function roundScore(value: number) {
  return Math.round(value * 1000) / 1000;
}

function average(left: number, right: number) {
  return (left + right) / 2;
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
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

function getVolatility(values: number[]) {
  const mean = getMean(values);
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;

  return Math.sqrt(variance);
}

function getWindowTrend(values: number[]) {
  if (values.length < 4) {
    return 0;
  }

  const segmentSize = Math.max(1, Math.floor(values.length / 3));
  const first = values.slice(0, segmentSize);
  const last = values.slice(-segmentSize);

  return getMean(last) - getMean(first);
}

function getAngryTensionIndex(scores: Scores) {
  const smileAverage = average(scores.mouthSmileLeft, scores.mouthSmileRight);
  const frownAverage = average(scores.mouthFrownLeft, scores.mouthFrownRight);
  const browTension = average(scores.browDownLeft, scores.browDownRight);
  const eyeNarrowing = average(scores.eyeSquintLeft, scores.eyeSquintRight);
  const lowSmile = 1 - Math.max(0, Math.min(1, smileAverage));
  const synergy =
    browTension > 0.24 && eyeNarrowing > 0.18 && smileAverage < 0.24
      ? 0.22
      : 0;

  return clamp01(
    browTension * 0.42 +
      eyeNarrowing * 0.28 +
      frownAverage * 0.18 +
      lowSmile * 0.1 +
      synergy,
  );
}

function getScoreStats(samples: ScoreSample[], key: ScoreKey): ScoreStats {
  const values = samples.map((sample) => sample.scores[key]);
  const latest = values[values.length - 1] ?? 0;
  const mean = getMean(values);
  const max = Math.max(...values);
  const min = Math.min(...values);
  const volatility = getVolatility(values);

  return {
    latest: roundScore(latest),
    mean: roundScore(mean),
    max: roundScore(max),
    min: roundScore(min),
    delta: roundScore(latest - values[0]),
    volatility: roundScore(volatility),
    stability: roundScore(clamp01(1 - volatility / 0.22)),
  };
}

function getTemporalFeatures(samples: ScoreSample[]): TemporalFeatures {
  const scoreStats = Object.fromEntries(
    SCORE_KEYS.map((key) => [key, getScoreStats(samples, key)]),
  ) as Record<ScoreKey, ScoreStats>;
  const smartScores = Object.fromEntries(
    SCORE_KEYS.map((key) => {
      const stats = scoreStats[key];

      return [
        key,
        roundScore(stats.latest * 0.45 + stats.mean * 0.35 + stats.max * 0.2),
      ];
    }),
  ) as Scores;
  const firstTimestamp = samples[0]?.timestamp ?? 0;
  const lastTimestamp = samples[samples.length - 1]?.timestamp ?? firstTimestamp;
  const angrySeries = samples.map((sample) => getAngryTensionIndex(sample.scores));
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
    sampleCount: samples.length,
    windowMs: roundScore(lastTimestamp - firstTimestamp),
    scoreStats,
    smartScores,
    aggregate: {
      angryTensionMean: roundScore(getMean(angrySeries)),
      angryTensionPeak: roundScore(Math.max(...angrySeries)),
      angryTensionTrend: roundScore(getWindowTrend(angrySeries)),
      angryTensionStableRatio: roundScore(
        angrySeries.filter((value) => value >= 0.45).length / angrySeries.length,
      ),
      smileMean: roundScore(getMean(smileSeries)),
      smilePeak: roundScore(Math.max(...smileSeries)),
      browTensionMean: roundScore(getMean(browSeries)),
      browTensionPeak: roundScore(Math.max(...browSeries)),
      eyeNarrowingMean: roundScore(getMean(eyeSeries)),
      frownMean: roundScore(getMean(frownSeries)),
    },
  };
}

function getExpressionScores(params: {
  smileAverage: number;
  frownAverage: number;
  browTension: number;
  eyeNarrowing: number;
  jawOpening: number;
  innerBrowLift: number;
  activationLevel: number;
}) {
  const smile = normalizeRange(params.smileAverage, 0.16, 0.52);
  const frown = normalizeRange(params.frownAverage, 0.1, 0.42);
  const brow = normalizeRange(params.browTension, 0.12, 0.44);
  const squint = normalizeRange(params.eyeNarrowing, 0.1, 0.4);
  const jaw = normalizeRange(params.jawOpening, 0.14, 0.55);
  const innerBrow = normalizeRange(params.innerBrowLift, 0.12, 0.45);
  const activation = normalizeRange(params.activationLevel, 0.1, 0.45);
  const lowSmile = 1 - smile;
  const angrySynergy =
    params.browTension > 0.24 &&
    params.eyeNarrowing > 0.18 &&
    params.smileAverage < 0.24
      ? 0.22
      : 0;

  return {
    happy: roundScore(
      clamp01(0.72 * smile + 0.12 * squint + 0.1 * activation - 0.22 * frown - 0.18 * brow),
    ),
    sad: roundScore(
      clamp01(0.42 * frown + 0.35 * innerBrow + 0.16 * lowSmile - 0.12 * brow - 0.08 * smile),
    ),
    angry: roundScore(
      clamp01(0.42 * brow + 0.28 * squint + 0.18 * frown + 0.1 * lowSmile + angrySynergy - 0.16 * smile),
    ),
    surprised: roundScore(
      clamp01(0.46 * jaw + 0.34 * innerBrow + 0.12 * lowSmile - 0.16 * squint - 0.12 * brow),
    ),
    tired: roundScore(
      clamp01(0.42 * squint + 0.18 * innerBrow + 0.16 * lowSmile - 0.18 * brow - 0.12 * jaw),
    ),
    neutral: roundScore(clamp01(1 - normalizeRange(params.activationLevel, 0.08, 0.32))),
    unclear: 0,
  } satisfies Record<Expression, number>;
}

function getHeuristicClassification(
  expressionScores: Record<Expression, number>,
  evidence: string[],
  activationLevel: number,
) {
  const ranked = (Object.entries(expressionScores) as Array<[Expression, number]>)
    .filter(([expression]) => expression !== "unclear")
    .sort((a, b) => b[1] - a[1]);
  const [topExpression, topScore] = ranked[0] ?? ["unclear", 0];
  const secondScore = ranked[1]?.[1] ?? 0;
  const margin = topScore - secondScore;

  if (activationLevel < 0.12) {
    return {
      expression: "neutral" as const,
      confidence: 0.72,
      evidence: ["整体表情激活度很低，优先视为可见中性表情"],
    };
  }

  if (topScore < 0.32 || margin < 0.06) {
    return {
      expression: "unclear" as const,
      confidence: roundScore(0.35 + topScore * 0.25),
      evidence: [
        "最高候选分数不高或候选之间差距很小，表情信号冲突",
        ...evidence.slice(0, 3),
      ],
    };
  }

  return {
    expression: topExpression,
    confidence: roundScore(clamp01(0.42 + topScore * 0.38 + margin * 0.5)),
    evidence: evidence.slice(0, 5),
  };
}

function getDerivedFeatures(scores: Scores, samples: ScoreSample[]): DerivedFeatures {
  const temporalFeatures = getTemporalFeatures(samples);
  const featureScores = temporalFeatures.smartScores;
  const smileAverage = average(
    featureScores.mouthSmileLeft,
    featureScores.mouthSmileRight,
  );
  const frownAverage = average(
    featureScores.mouthFrownLeft,
    featureScores.mouthFrownRight,
  );
  const browTension = average(
    featureScores.browDownLeft,
    featureScores.browDownRight,
  );
  const eyeNarrowing = average(
    featureScores.eyeSquintLeft,
    featureScores.eyeSquintRight,
  );
  const activationLevel = Math.max(
    smileAverage,
    frownAverage,
    browTension,
    eyeNarrowing,
    featureScores.jawOpen,
    featureScores.browInnerUp,
  );
  const evidence: string[] = [];
  const heuristicHints: string[] = [];

  if (smileAverage > 0.35 && frownAverage < 0.2) {
    heuristicHints.push("smile-dominant visible expression");
    evidence.push("左右嘴角上扬均值较高，且嘴角下压信号较低");
  }

  if (frownAverage > 0.25 || featureScores.browInnerUp > 0.35) {
    heuristicHints.push("downturned mouth or raised inner brow signal");
    evidence.push("嘴角下压或内眉抬起较明显");
  }

  if (browTension > 0.25 && eyeNarrowing > 0.2) {
    heuristicHints.push("brow tension with narrowed eyes");
    evidence.push("眉部压低和眼部收窄同时升高，是生气/紧绷类可见信号");
  }

  if (browTension > 0.32 && eyeNarrowing > 0.22 && smileAverage < 0.24) {
    heuristicHints.push("strong angry-like tension pattern");
    evidence.push("眉压低强、眼收窄明显、微笑信号弱，符合 angry-like tension 组合");
  }

  if (frownAverage > 0.2 && smileAverage < 0.18 && browTension > 0.2) {
    heuristicHints.push("frown plus brow tension without smile");
    evidence.push("嘴角下压叠加眉部压低，同时缺少嘴角上扬");
  }

  if (featureScores.jawOpen > 0.28 && featureScores.browInnerUp > 0.2) {
    heuristicHints.push("open jaw with raised inner brow");
    evidence.push("下颌张开和内眉抬起同时出现，可能偏惊讶");
  }

  if (activationLevel < 0.18) {
    heuristicHints.push("low activation; prefer neutral or unclear");
    evidence.push("整体激活度较低，应降低非中性表情置信度");
  }

  if (
    temporalFeatures.aggregate.angryTensionMean > 0.42 &&
    temporalFeatures.aggregate.angryTensionStableRatio > 0.35
  ) {
    heuristicHints.push("temporally stable angry-like tension");
    evidence.push(
      "最近时间窗口内 angry/tension 指数持续偏高，而不是单帧尖峰",
    );
  }

  if (temporalFeatures.aggregate.angryTensionTrend > 0.12) {
    heuristicHints.push("rising angry-like tension trend");
    evidence.push("最近几秒 angry/tension 指数呈上升趋势");
  }

  const expressionScores = getExpressionScores({
    smileAverage,
    frownAverage,
    browTension,
    eyeNarrowing,
    jawOpening: featureScores.jawOpen,
    innerBrowLift: featureScores.browInnerUp,
    activationLevel,
  });

  expressionScores.angry = roundScore(
    clamp01(
      expressionScores.angry * 0.72 +
        temporalFeatures.aggregate.angryTensionMean * 0.18 +
        temporalFeatures.aggregate.angryTensionStableRatio * 0.18 +
        Math.max(0, temporalFeatures.aggregate.angryTensionTrend) * 0.3,
    ),
  );

  const heuristicClassification = getHeuristicClassification(
    expressionScores,
    evidence,
    activationLevel,
  );

  return {
    smileAverage: roundScore(smileAverage),
    frownAverage: roundScore(frownAverage),
    browTension: roundScore(browTension),
    eyeNarrowing: roundScore(eyeNarrowing),
    jawOpening: roundScore(featureScores.jawOpen),
    innerBrowLift: roundScore(featureScores.browInnerUp),
    smileFrownBalance: roundScore(smileAverage - frownAverage),
    leftRightSmileAsymmetry: roundScore(
      Math.abs(scores.mouthSmileLeft - scores.mouthSmileRight),
    ),
    activationLevel: roundScore(activationLevel),
    temporalFeatures,
    strongestScores: [...SCORE_KEYS]
      .map((name) => ({ name, score: roundScore(scores[name]) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5),
    expressionScores,
    heuristicClassification,
    heuristicHints,
  };
}

function maybeApplyRuleCorrection(
  analysis: AnalysisResponse,
  derivedFeatures: DerivedFeatures,
): AnalysisResponse {
  const heuristic = derivedFeatures.heuristicClassification;
  const temporalAngry =
    derivedFeatures.temporalFeatures.aggregate.angryTensionMean >= 0.48 &&
    derivedFeatures.temporalFeatures.aggregate.angryTensionStableRatio >= 0.35;
  const peakAngry =
    derivedFeatures.temporalFeatures.aggregate.angryTensionPeak >= 0.72 &&
    derivedFeatures.expressionScores.angry >= 0.58;

  if (
    ((heuristic.expression === "angry" && heuristic.confidence >= 0.62) ||
      temporalAngry ||
      peakAngry) &&
    analysis.expression !== "angry"
  ) {
    return {
      expression: "angry",
      confidence: Math.max(analysis.confidence, heuristic.confidence, 0.72),
      reason: `规则层检测到更强的可见生气/紧绷组合：${heuristic.evidence.join("；")}。时间窗口 angry/tension 均值=${derivedFeatures.temporalFeatures.aggregate.angryTensionMean}，峰值=${derivedFeatures.temporalFeatures.aggregate.angryTensionPeak}，稳定比例=${derivedFeatures.temporalFeatures.aggregate.angryTensionStableRatio}。因此相较于模型原始候选 ${analysis.expression}，当前更适合标记为 angry-like visible expression。`,
      warning: WARNING,
    };
  }

  if (
    heuristic.expression !== "unclear" &&
    heuristic.confidence >= 0.74 &&
    analysis.expression === "unclear"
  ) {
    return {
      expression: heuristic.expression,
      confidence: heuristic.confidence,
      reason: `规则层给出了更稳定的可见表情候选：${heuristic.evidence.join("；")}。`,
      warning: WARNING,
    };
  }

  return analysis;
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

function getOpenAIErrorMessage(error: unknown) {
  if (!isRecord(error)) {
    return "Unable to analyze expression.";
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

  return "Unable to analyze expression.";
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
    const samples = normalizeScoreSamples(
      isRecord(body) ? body.samples : undefined,
      scores,
    );
    const derivedFeatures = getDerivedFeatures(scores, samples);

    const completion = await client.chat.completions.create({
      model: "gpt-5.5",
      messages: [
        {
          role: "developer",
          content:
            "Analyze only the visible facial expression based on the provided MediaPipe face blendshape scores, recent temporal score window, expression score breakdown, heuristic classification, and derived feature summary. Treat the heuristic classification as an expert prior, especially for angry-like visible tension: sustained browDownLeft/browDownRight plus eyeSquintLeft/eyeSquintRight, low smile, and mouth frown should strongly support angry when active together. Prefer temporally stable patterns over one-frame spikes. Compare active and inactive signals, handle conflicting cues, and lower confidence when signals are weak or contradictory. Do not identify the person. Do not infer race, gender, age, health, mental health, or any sensitive attributes. Do not claim to know the user's real mood. Do not diagnose emotional or psychological state. Return strict JSON only with the requested format. Write the reason in Simplified Chinese and mention the most important visible score patterns.",
        },
        {
          role: "user",
          content: JSON.stringify({
            scores,
            samples,
            derivedFeatures,
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

    const analysis = maybeApplyRuleCorrection(
      normalizeAnalysis(JSON.parse(content)),
      derivedFeatures,
    );

    return NextResponse.json(analysis);
  } catch (error: unknown) {
    console.error("Expression analysis failed:", error);

    return NextResponse.json(
      { error: getOpenAIErrorMessage(error) },
      { status: 502 },
    );
  }
}
