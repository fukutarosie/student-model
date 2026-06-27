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

const WARNING =
  "This is only an expression estimate, not a real mood or mental health diagnosis.";

type ScoreKey = (typeof SCORE_KEYS)[number];
type Scores = Record<ScoreKey, number>;
type NumericScores = Record<string, number>;
type Expression = (typeof EXPRESSIONS)[number];

type ScoreSample = {
  timestamp: number;
  scores: Scores;
  allScores: NumericScores;
  visionMetrics: unknown;
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

type ReportAlternative = {
  expression: Expression;
  confidence: number;
  reason: string;
};

type SignalHighlight = {
  name: string;
  score: number;
  note: string;
};

type ExpressionReport = {
  summary: string;
  primaryCues: string[];
  counterSignals: string[];
  alternatives: ReportAlternative[];
  temporalNotes: string[];
  signalHighlights: SignalHighlight[];
};

type AnalysisResponse = {
  expression: Expression;
  confidence: number;
  reason: string;
  warning: string;
  report: ExpressionReport;
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

function normalizeNumericScores(value: unknown, fallback: NumericScores = {}) {
  if (!isRecord(value)) {
    return fallback;
  }

  const scores: NumericScores = {};

  for (const [key, score] of Object.entries(value)) {
    if (typeof score === "number" && Number.isFinite(score)) {
      scores[key] = Math.max(0, Math.min(1, score));
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

function normalizeScoreSamples(
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

  for (const item of value) {
    if (!isRecord(item) || typeof item.timestamp !== "number") {
      continue;
    }

    const scores = normalizeScores(item.scores);

    if (!scores) {
      continue;
    }

    const allScores = normalizeNumericScores(item.allScores, fallbackScores);

    samples.push({
      timestamp: item.timestamp,
      scores,
      allScores,
      visionMetrics: sanitizeJsonValue(item.visionMetrics) ?? null,
    });
  }

  if (samples.length === 0) {
    return [
      {
        timestamp: 0,
        scores: fallbackScores,
        allScores: fallbackAllScores,
        visionMetrics: null,
      },
    ];
  }

  const sortedSamples = samples
    .sort((left, right) => left.timestamp - right.timestamp)
    .slice(-MAX_SCORE_SAMPLES);
  const latestTimestamp =
    sortedSamples[sortedSamples.length - 1]?.timestamp ??
    sortedSamples[0].timestamp;
  const windowedSamples = sortedSamples.filter(
    (sample) => latestTimestamp - sample.timestamp <= MAX_SAMPLE_WINDOW_MS,
  );

  return windowedSamples.length > 0
    ? windowedSamples
    : [
        {
          timestamp: 0,
          scores: fallbackScores,
          allScores: fallbackAllScores,
          visionMetrics: null,
        },
      ];
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
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    values.length;

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
  const browCore = normalizeRange(browTension, 0.12, 0.44);
  const eyeCore = normalizeRange(eyeNarrowing, 0.1, 0.4);
  const frownCore = normalizeRange(frownAverage, 0.1, 0.42);
  const lowSmileWeakCue = normalizeRange(0.24 - smileAverage, 0, 0.24);
  const jawOpen = normalizeRange(scores.jawOpen, 0.18, 0.56);
  const innerBrowLift = normalizeRange(scores.browInnerUp, 0.16, 0.48);
  const browEyeAgreement = Math.min(browCore, eyeCore);
  const synergy =
    browTension > 0.26 &&
    eyeNarrowing > 0.2 &&
    (frownAverage > 0.14 || smileAverage < 0.12)
      ? 0.14
      : 0;

  return clamp01(
    browEyeAgreement * 0.46 +
      browCore * 0.22 +
      eyeCore * 0.18 +
      frownCore * 0.2 +
      lowSmileWeakCue * 0.04 +
      synergy -
      jawOpen * 0.1 -
      innerBrowLift * 0.08,
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
  const angrySeries = samples.map((sample) =>
    getAngryTensionIndex(sample.scores),
  );
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
        angrySeries.filter((value) => value >= 0.45).length /
          angrySeries.length,
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
  const lowJaw = 1 - jaw;
  const browEyeAgreement = Math.min(brow, squint);
  const angrySynergy =
    params.browTension > 0.26 &&
    params.eyeNarrowing > 0.2 &&
    (params.frownAverage > 0.14 || params.smileAverage < 0.12)
      ? 0.16
      : 0;

  return {
    happy: roundScore(
      clamp01(
        0.72 * smile +
          0.12 * squint +
          0.1 * activation -
          0.22 * frown -
          0.18 * brow,
      ),
    ),
    neutral: roundScore(
      clamp01(1 - normalizeRange(params.activationLevel, 0.08, 0.32)),
    ),
    focused: roundScore(
      clamp01(
        0.34 * squint +
          0.24 * lowSmile +
          0.18 * lowJaw +
          0.16 * activation -
          0.16 * frown -
          0.14 * innerBrow,
      ),
    ),
    serious: roundScore(
      clamp01(
        0.28 * lowSmile +
          0.22 * frown +
          0.2 * brow +
          0.14 * lowJaw -
          0.18 * smile -
          0.12 * jaw,
      ),
    ),
    sad: roundScore(
      clamp01(
        0.42 * frown +
          0.35 * innerBrow +
          0.16 * lowSmile -
          0.12 * brow -
          0.08 * smile,
      ),
    ),
    angry: roundScore(
      clamp01(
        0.36 * browEyeAgreement +
          0.22 * brow +
          0.18 * squint +
          0.2 * frown +
          0.04 * lowSmile +
          angrySynergy -
          0.16 * smile -
          0.1 * innerBrow -
          0.1 * jaw,
      ),
    ),
    surprised: roundScore(
      clamp01(
        0.46 * jaw +
          0.34 * innerBrow +
          0.12 * lowSmile -
          0.16 * squint -
          0.12 * brow,
      ),
    ),
    tired: roundScore(
      clamp01(
        0.42 * squint +
          0.18 * innerBrow +
          0.16 * lowSmile -
          0.18 * brow -
          0.12 * jaw,
      ),
    ),
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
      evidence: [
        "Overall visible activation is very low, so neutral is favored.",
      ],
    };
  }

  if (topScore < 0.32 || margin < 0.06) {
    return {
      expression: "unclear" as const,
      confidence: roundScore(0.35 + topScore * 0.25),
      evidence: [
        "The leading expression score is weak or close to another candidate.",
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
    evidence.push("Smile scores are high while mouth-frown scores are low.");
  }

  if (frownAverage > 0.25 || featureScores.browInnerUp > 0.35) {
    heuristicHints.push("downturned mouth or raised inner brow signal");
    evidence.push("Mouth-frown or inner-brow lift is visible.");
  }

  if (browTension > 0.25 && eyeNarrowing > 0.2) {
    heuristicHints.push("brow tension with narrowed eyes");
    evidence.push("Brow-down and eye-squint scores are elevated together.");
  }

  if (
    browTension > 0.32 &&
    eyeNarrowing > 0.22 &&
    (frownAverage > 0.14 || smileAverage < 0.12)
  ) {
    heuristicHints.push("strong angry-like tension pattern");
    evidence.push(
      "Strong brow-down and eye-squint appear with frown or mouth tension.",
    );
  }

  if (frownAverage > 0.2 && smileAverage < 0.18 && browTension > 0.2) {
    heuristicHints.push("frown plus brow tension without smile");
    evidence.push("Mouth-frown combines with brow tension and low smile.");
  }

  if (featureScores.jawOpen > 0.28 && featureScores.browInnerUp > 0.2) {
    heuristicHints.push("open jaw with raised inner brow");
    evidence.push("Open jaw and raised inner brow point toward surprise.");
  }

  if (activationLevel < 0.18) {
    heuristicHints.push("low activation; prefer neutral or unclear");
    evidence.push("Overall activation is low; non-neutral confidence is reduced.");
  }

  if (
    temporalFeatures.aggregate.angryTensionMean > 0.42 &&
    temporalFeatures.aggregate.angryTensionStableRatio > 0.35
  ) {
    heuristicHints.push("temporally stable angry-like tension");
    evidence.push(
      "Angry/tension index stays elevated across the recent time window.",
    );
  }

  if (temporalFeatures.aggregate.angryTensionTrend > 0.12) {
    heuristicHints.push("rising angry-like tension trend");
    evidence.push("Angry/tension index is rising in the recent window.");
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
      expressionScores.angry * 0.78 +
        temporalFeatures.aggregate.angryTensionMean * 0.14 +
        temporalFeatures.aggregate.angryTensionStableRatio * 0.08 +
        Math.max(0, temporalFeatures.aggregate.angryTensionTrend) * 0.2,
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

function getVisibleEvidenceSummary(derivedFeatures: DerivedFeatures) {
  const aggregate = derivedFeatures.temporalFeatures.aggregate;

  return [
    `smile=${derivedFeatures.smileAverage}`,
    `frown=${derivedFeatures.frownAverage}`,
    `browDown=${derivedFeatures.browTension}`,
    `eyeSquint=${derivedFeatures.eyeNarrowing}`,
    `jawOpen=${derivedFeatures.jawOpening}`,
    `browInnerUp=${derivedFeatures.innerBrowLift}`,
    `angryTensionMean=${aggregate.angryTensionMean}`,
    `angryTensionStableRatio=${aggregate.angryTensionStableRatio}`,
  ];
}

function getModelDerivedFeatures(derivedFeatures: DerivedFeatures) {
  return {
    ...derivedFeatures,
    heuristicClassification: {
      ...derivedFeatures.heuristicClassification,
      evidence: getVisibleEvidenceSummary(derivedFeatures),
    },
    analysisMode:
      "OpenAI is the final classifier. Local scores and heuristics are descriptive context only.",
    reportInstructions: {
      summary:
        "Provide a careful visible-expression report without claiming true mood.",
      alternatives:
        "Compare the main result against plausible alternatives such as focused, serious, tired, angry, sad, surprised, neutral, and unclear.",
    },
  };
}

function getString(value: unknown, fallback: string) {
  return typeof value === "string" && value.trim().length > 0
    ? value
    : fallback;
}

function getStringArray(value: unknown, fallback: string[], maxItems = 6) {
  if (!Array.isArray(value)) {
    return fallback;
  }

  const strings = value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, maxItems);

  return strings.length > 0 ? strings : fallback;
}

function getExpression(value: unknown): Expression {
  return EXPRESSIONS.includes(value as Expression)
    ? (value as Expression)
    : "unclear";
}

function getReportAlternatives(value: unknown): ReportAlternative[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter(isRecord)
    .map((item) => ({
      expression: getExpression(item.expression),
      confidence: clampConfidence(item.confidence),
      reason: getString(item.reason, "No alternative reason provided."),
    }))
    .slice(0, 4);
}

function getSignalHighlights(value: unknown): SignalHighlight[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter(isRecord)
    .map((item) => {
      return {
        name: getString(item.name, "unknownSignal"),
        score: clampConfidence(item.score),
        note: getString(item.note, "Visible score contributed to the report."),
      };
    })
    .slice(0, 6);
}

function normalizeReport(value: unknown, reason: string): ExpressionReport {
  const report = isRecord(value) ? value : {};

  return {
    summary: getString(report.summary, reason),
    primaryCues: getStringArray(report.primaryCues, [reason], 5),
    counterSignals: getStringArray(report.counterSignals, [], 4),
    alternatives: getReportAlternatives(report.alternatives),
    temporalNotes: getStringArray(report.temporalNotes, [], 3),
    signalHighlights: getSignalHighlights(report.signalHighlights),
  };
}

function normalizeAnalysis(value: unknown): AnalysisResponse {
  if (!isRecord(value)) {
    throw new Error("OpenAI returned an invalid response.");
  }

  const expression = getExpression(value.expression);
  const reason = getString(
    value.reason,
    "The visible facial expression scores were unclear.",
  );

  return {
    expression,
    confidence: clampConfidence(value.confidence),
    reason,
    warning: WARNING,
    report: normalizeReport(value.report, reason),
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
    const allScores = normalizeNumericScores(
      isRecord(body) ? body.allScores : undefined,
      scores,
    );
    const visionMetrics =
      sanitizeJsonValue(isRecord(body) ? body.visionMetrics : undefined) ??
      null;
    const samples = normalizeScoreSamples(
      isRecord(body) ? body.samples : undefined,
      scores,
      allScores,
    );
    const derivedFeatures = getDerivedFeatures(scores, samples);

    const completion = await client.chat.completions.create({
      model: "gpt-5.5",
      messages: [
        {
          role: "developer",
          content:
            "Analyze only the visible facial expression based on numeric browser-local vision signals: core MediaPipe blendshape scores, complete MediaPipe blendshape scores, landmark-derived geometry, facial transformation matrix head-pose estimates, local frame quality metrics, recent temporal samples, expression score breakdown, heuristic classification, and derived feature summary. You are the final classifier; local heuristic scores are descriptive context, not rules to obey. Return a careful structured report based only on these visible-expression numbers. Focused and serious are allowed labels, and should be used when a face looks concentrated, stern, or unsmiling without enough evidence for tired, angry, sad, or neutral. Angry can appear through several visible patterns, not only narrowed eyes: brow-down tension, eye narrowing, frown or mouth compression, jaw opening, asymmetry, head pose, and temporal stability may all matter depending on the combination. Do not require one fixed angry template. Do not classify from one isolated cue such as low smile, eye squint, or frown alone; compare active and inactive signals and choose the most plausible visible expression. Use frame quality and face pose to lower confidence when the face is poorly framed, dim, blurry, far away, off-center, or turned. Distinguish angry from tired squint, focused attention, serious unsmiling expression, sad frown plus browInnerUp, and surprised jawOpen plus browInnerUp. Prefer temporally stable patterns over one-frame spikes. Lower confidence when signals are weak or contradictory. Do not identify the person. Do not infer race, gender, age, health, mental health, or any sensitive attributes. Do not claim to know the user's real mood. Do not diagnose emotional or psychological state. Do not mention uploaded images or video; only numeric features were provided. Return strict JSON only with the requested format. Write all report text in English.",
        },
        {
          role: "user",
          content: JSON.stringify({
            scores,
            allBlendshapeScores: allScores,
            visionMetrics,
            samples,
            derivedFeatures: getModelDerivedFeatures(derivedFeatures),
            availableSignalGroups: [
              "core blendshape scores",
              "complete MediaPipe blendshape scores",
              "landmark-derived geometry metrics",
              "head pose from facial transformation matrix",
              "frame quality metrics from local camera pixels",
              "recent temporal samples",
            ],
            allowedExpressions: EXPRESSIONS,
            requiredWarning: WARNING,
          }),
        },
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "visible_expression_report",
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
                  "One short reason based only on visible facial expression scores.",
              },
              warning: {
                type: "string",
                enum: [WARNING],
              },
              report: {
                type: "object",
                additionalProperties: false,
                properties: {
                  summary: {
                    type: "string",
                    description:
                      "Concise visible-expression conclusion in one or two sentences.",
                  },
                  primaryCues: {
                    type: "array",
                    minItems: 2,
                    maxItems: 5,
                    items: {
                      type: "string",
                    },
                  },
                  counterSignals: {
                    type: "array",
                    minItems: 1,
                    maxItems: 4,
                    items: {
                      type: "string",
                    },
                  },
                  alternatives: {
                    type: "array",
                    minItems: 2,
                    maxItems: 4,
                    items: {
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
                        },
                      },
                      required: ["expression", "confidence", "reason"],
                    },
                  },
                  temporalNotes: {
                    type: "array",
                    minItems: 1,
                    maxItems: 3,
                    items: {
                      type: "string",
                    },
                  },
                  signalHighlights: {
                    type: "array",
                    minItems: 3,
                    maxItems: 6,
                    items: {
                      type: "object",
                      additionalProperties: false,
                      properties: {
                        name: {
                          type: "string",
                          description:
                            "Blendshape, landmark-derived, pose, or frame-quality signal name.",
                        },
                        score: {
                          type: "number",
                          minimum: 0,
                          maximum: 1,
                        },
                        note: {
                          type: "string",
                        },
                      },
                      required: ["name", "score", "note"],
                    },
                  },
                },
                required: [
                  "summary",
                  "primaryCues",
                  "counterSignals",
                  "alternatives",
                  "temporalNotes",
                  "signalHighlights",
                ],
              },
            },
            required: ["expression", "confidence", "reason", "warning", "report"],
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
  } catch (error: unknown) {
    console.error("Expression analysis failed:", error);

    return NextResponse.json(
      { error: getOpenAIErrorMessage(error) },
      { status: 502 },
    );
  }
}
