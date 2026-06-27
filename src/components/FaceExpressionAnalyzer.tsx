"use client";

import {
  FaceLandmarker,
  FilesetResolver,
  type Category,
  type FaceLandmarkerResult,
  type Matrix,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const CORE_BLENDSHAPE_KEYS = [
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

type CoreBlendshapeKey = (typeof CORE_BLENDSHAPE_KEYS)[number];
type CoreBlendshapeScores = Record<CoreBlendshapeKey, number>;
type AllBlendshapeScores = Record<string, number>;

type FaceBounds = {
  x: number;
  y: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
  areaRatio: number;
  centerOffset: number;
};

type LandmarkMetrics = {
  landmarkCount: number;
  eyeOpenness: number;
  mouthOpening: number;
  mouthWidth: number;
  browEyeDistance: number;
  eyeDistance: number;
  faceYawBalance: number;
  faceVerticalTilt: number;
  depthRange: number;
  geometryAsymmetry: number;
};

type HeadPose = {
  yaw: number;
  pitch: number;
  roll: number;
  matrixAvailable: boolean;
};

type FrameMetrics = {
  width: number;
  height: number;
  aspectRatio: number;
  brightness: number;
  contrast: number;
  sharpness: number;
  faceAreaRatio: number;
  centerOffset: number;
  qualityScore: number;
};

type CameraSettings = {
  width?: number;
  height?: number;
  frameRate?: number;
  aspectRatio?: number;
  facingMode?: string;
};

type VisionMetrics = {
  faceBounds: FaceBounds;
  landmarkMetrics: LandmarkMetrics;
  headPose: HeadPose;
  frameMetrics: FrameMetrics;
  cameraSettings: CameraSettings;
};

type ScoreSample = {
  timestamp: number;
  scores: CoreBlendshapeScores;
  allScores: AllBlendshapeScores;
  visionMetrics: VisionMetrics | null;
};

type Expression =
  | "happy"
  | "neutral"
  | "focused"
  | "serious"
  | "sad"
  | "angry"
  | "surprised"
  | "tired"
  | "unclear";

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

type AnalysisResult = {
  expression: Expression;
  confidence: number;
  reason: string;
  warning: string;
  report: ExpressionReport;
};

type SignalRow = {
  label: string;
  detail: string;
  value: number;
};

const EXPRESSION_LABELS: Record<Expression, string> = {
  happy: "Happy",
  neutral: "Neutral",
  focused: "Focused",
  serious: "Serious",
  sad: "Sad",
  angry: "Angry / Tense",
  surprised: "Surprised",
  tired: "Tired",
  unclear: "Unclear",
};

const BLENDSHAPE_LABELS: Record<CoreBlendshapeKey, string> = {
  mouthSmileLeft: "Left smile",
  mouthSmileRight: "Right smile",
  browDownLeft: "Left brow down",
  browDownRight: "Right brow down",
  eyeSquintLeft: "Left eye squint",
  eyeSquintRight: "Right eye squint",
  jawOpen: "Jaw open",
  browInnerUp: "Inner brow up",
  mouthFrownLeft: "Left frown",
  mouthFrownRight: "Right frown",
};

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const SAMPLE_WINDOW_MS = 4000;
const SAMPLE_INTERVAL_MS = 120;
const MAX_SCORE_SAMPLES = 48;
const FRAME_SAMPLE_WIDTH = 56;
const FRAME_SAMPLE_HEIGHT = 40;

function isMediaPipeInfoLog(args: unknown[]) {
  return args.some(
    (arg) =>
      typeof arg === "string" &&
      arg.includes("Created TensorFlow Lite XNNPACK delegate for CPU"),
  );
}

function createEmptyScores(): CoreBlendshapeScores {
  return Object.fromEntries(
    CORE_BLENDSHAPE_KEYS.map((key) => [key, 0]),
  ) as CoreBlendshapeScores;
}

function extractAllScores(categories: Category[]): AllBlendshapeScores {
  return Object.fromEntries(
    categories
      .filter((category) => Number.isFinite(category.score))
      .map((category) => [
        category.categoryName,
        Math.max(0, Math.min(1, category.score)),
      ]),
  );
}

function extractCoreScores(allScores: AllBlendshapeScores): CoreBlendshapeScores {
  const scores = createEmptyScores();

  for (const key of CORE_BLENDSHAPE_KEYS) {
    scores[key] = allScores[key] ?? 0;
  }

  return scores;
}

function average(left: number, right: number) {
  return (left + right) / 2;
}

function clampPercent(value: number) {
  return `${Math.max(0, Math.min(100, Math.round(value * 100)))}%`;
}

function formatConfidence(value: number) {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function roundMetric(value: number) {
  return Math.round(value * 1000) / 1000;
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function getDistance(
  left: NormalizedLandmark | undefined,
  right: NormalizedLandmark | undefined,
) {
  if (!left || !right) {
    return 0;
  }

  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function getLandmark(landmarks: NormalizedLandmark[], index: number) {
  return landmarks[index];
}

function getFaceBounds(landmarks: NormalizedLandmark[]): FaceBounds {
  const xs = landmarks.map((landmark) => landmark.x);
  const ys = landmarks.map((landmark) => landmark.y);
  const minX = Math.max(0, Math.min(...xs));
  const maxX = Math.min(1, Math.max(...xs));
  const minY = Math.max(0, Math.min(...ys));
  const maxY = Math.min(1, Math.max(...ys));
  const width = Math.max(0, maxX - minX);
  const height = Math.max(0, maxY - minY);
  const centerX = minX + width / 2;
  const centerY = minY + height / 2;
  const centerOffset = clamp01(Math.hypot(centerX - 0.5, centerY - 0.5) / 0.5);

  return {
    x: roundMetric(minX),
    y: roundMetric(minY),
    width: roundMetric(width),
    height: roundMetric(height),
    centerX: roundMetric(centerX),
    centerY: roundMetric(centerY),
    areaRatio: roundMetric(width * height),
    centerOffset: roundMetric(centerOffset),
  };
}

function getLandmarkMetrics(landmarks: NormalizedLandmark[]): LandmarkMetrics {
  const leftEyeOuter = getLandmark(landmarks, 33);
  const leftEyeInner = getLandmark(landmarks, 133);
  const rightEyeInner = getLandmark(landmarks, 362);
  const rightEyeOuter = getLandmark(landmarks, 263);
  const leftEyeUpper = getLandmark(landmarks, 159);
  const leftEyeLower = getLandmark(landmarks, 145);
  const rightEyeUpper = getLandmark(landmarks, 386);
  const rightEyeLower = getLandmark(landmarks, 374);
  const mouthLeft = getLandmark(landmarks, 61);
  const mouthRight = getLandmark(landmarks, 291);
  const mouthUpper = getLandmark(landmarks, 13);
  const mouthLower = getLandmark(landmarks, 14);
  const noseTip = getLandmark(landmarks, 1);
  const chin = getLandmark(landmarks, 152);
  const forehead = getLandmark(landmarks, 10);
  const leftBrow = getLandmark(landmarks, 105);
  const rightBrow = getLandmark(landmarks, 334);

  const leftEyeWidth = getDistance(leftEyeOuter, leftEyeInner);
  const rightEyeWidth = getDistance(rightEyeInner, rightEyeOuter);
  const leftEyeOpen = getDistance(leftEyeUpper, leftEyeLower);
  const rightEyeOpen = getDistance(rightEyeUpper, rightEyeLower);
  const leftEyeRatio = leftEyeWidth ? leftEyeOpen / leftEyeWidth : 0;
  const rightEyeRatio = rightEyeWidth ? rightEyeOpen / rightEyeWidth : 0;
  const mouthWidth = getDistance(mouthLeft, mouthRight);
  const mouthOpening = mouthWidth
    ? getDistance(mouthUpper, mouthLower) / mouthWidth
    : 0;
  const eyeDistance = getDistance(leftEyeInner, rightEyeInner);
  const eyeCenterX =
    leftEyeInner && rightEyeInner ? (leftEyeInner.x + rightEyeInner.x) / 2 : 0.5;
  const faceYawBalance = noseTip ? noseTip.x - eyeCenterX : 0;
  const faceVerticalTilt =
    forehead && chin && noseTip
      ? (noseTip.y - forehead.y) / Math.max(0.001, chin.y - forehead.y)
      : 0;
  const browEyeDistance = leftBrow && rightBrow && leftEyeUpper && rightEyeUpper
    ? average(Math.abs(leftBrow.y - leftEyeUpper.y), Math.abs(rightBrow.y - rightEyeUpper.y))
    : 0;
  const zs = landmarks.map((landmark) => landmark.z);
  const geometryAsymmetry = Math.abs(leftEyeRatio - rightEyeRatio);

  return {
    landmarkCount: landmarks.length,
    eyeOpenness: roundMetric(average(leftEyeRatio, rightEyeRatio)),
    mouthOpening: roundMetric(mouthOpening),
    mouthWidth: roundMetric(mouthWidth),
    browEyeDistance: roundMetric(browEyeDistance),
    eyeDistance: roundMetric(eyeDistance),
    faceYawBalance: roundMetric(faceYawBalance),
    faceVerticalTilt: roundMetric(faceVerticalTilt),
    depthRange: roundMetric(Math.max(...zs) - Math.min(...zs)),
    geometryAsymmetry: roundMetric(geometryAsymmetry),
  };
}

function getHeadPose(matrix: Matrix | undefined): HeadPose {
  const data = matrix?.data;

  if (!data || data.length < 16) {
    return {
      yaw: 0,
      pitch: 0,
      roll: 0,
      matrixAvailable: false,
    };
  }

  const yaw = Math.atan2(-data[2], data[0]);
  const pitch = Math.atan2(data[6], data[10]);
  const roll = Math.atan2(data[4], data[5]);

  return {
    yaw: roundMetric((yaw * 180) / Math.PI),
    pitch: roundMetric((pitch * 180) / Math.PI),
    roll: roundMetric((roll * 180) / Math.PI),
    matrixAvailable: true,
  };
}

function getCameraSettings(stream: MediaStream | null): CameraSettings {
  const settings = stream?.getVideoTracks()[0]?.getSettings();

  if (!settings) {
    return {};
  }

  return {
    width: settings.width,
    height: settings.height,
    frameRate: settings.frameRate,
    aspectRatio: settings.aspectRatio,
    facingMode:
      typeof settings.facingMode === "string" ? settings.facingMode : undefined,
  };
}

function getVideoTextureMetrics(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
) {
  try {
    canvas.width = FRAME_SAMPLE_WIDTH;
    canvas.height = FRAME_SAMPLE_HEIGHT;
    const context = canvas.getContext("2d", { willReadFrequently: true });

    if (!context) {
      return { brightness: 0, contrast: 0, sharpness: 0 };
    }

    context.drawImage(video, 0, 0, FRAME_SAMPLE_WIDTH, FRAME_SAMPLE_HEIGHT);
    const pixels = context.getImageData(
      0,
      0,
      FRAME_SAMPLE_WIDTH,
      FRAME_SAMPLE_HEIGHT,
    ).data;
    const luminance: number[] = [];

    for (let index = 0; index < pixels.length; index += 4) {
      luminance.push(
        (0.2126 * pixels[index] +
          0.7152 * pixels[index + 1] +
          0.0722 * pixels[index + 2]) /
          255,
      );
    }

    const mean =
      luminance.reduce((sum, value) => sum + value, 0) / luminance.length;
    const contrast = Math.sqrt(
      luminance.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
        luminance.length,
    );
    let edgeEnergy = 0;
    let edgeCount = 0;

    for (let y = 1; y < FRAME_SAMPLE_HEIGHT; y += 1) {
      for (let x = 1; x < FRAME_SAMPLE_WIDTH; x += 1) {
        const current = luminance[y * FRAME_SAMPLE_WIDTH + x];
        const left = luminance[y * FRAME_SAMPLE_WIDTH + x - 1];
        const top = luminance[(y - 1) * FRAME_SAMPLE_WIDTH + x];
        edgeEnergy += Math.abs(current - left) + Math.abs(current - top);
        edgeCount += 2;
      }
    }

    return {
      brightness: roundMetric(mean),
      contrast: roundMetric(contrast),
      sharpness: roundMetric(edgeCount ? edgeEnergy / edgeCount : 0),
    };
  } catch {
    return { brightness: 0, contrast: 0, sharpness: 0 };
  }
}

function getVisionMetrics(params: {
  landmarks: NormalizedLandmark[];
  matrix: Matrix | undefined;
  video: HTMLVideoElement;
  stream: MediaStream | null;
  canvas: HTMLCanvasElement;
}): VisionMetrics | null {
  if (params.landmarks.length === 0) {
    return null;
  }

  const faceBounds = getFaceBounds(params.landmarks);
  const texture = getVideoTextureMetrics(params.video, params.canvas);
  const width = params.video.videoWidth || 0;
  const height = params.video.videoHeight || 0;
  const qualityScore = clamp01(
    faceBounds.areaRatio * 3.2 +
      (1 - faceBounds.centerOffset) * 0.22 +
      texture.contrast * 1.4 +
      texture.sharpness * 3,
  );

  return {
    faceBounds,
    landmarkMetrics: getLandmarkMetrics(params.landmarks),
    headPose: getHeadPose(params.matrix),
    frameMetrics: {
      width,
      height,
      aspectRatio: height ? roundMetric(width / height) : 0,
      brightness: texture.brightness,
      contrast: texture.contrast,
      sharpness: texture.sharpness,
      faceAreaRatio: faceBounds.areaRatio,
      centerOffset: faceBounds.centerOffset,
      qualityScore: roundMetric(qualityScore),
    },
    cameraSettings: getCameraSettings(params.stream),
  };
}

function getSignalLabel(name: string) {
  return BLENDSHAPE_LABELS[name as CoreBlendshapeKey] ?? name;
}

function getAngryTensionIndex(scores: CoreBlendshapeScores) {
  const smileAverage = average(scores.mouthSmileLeft, scores.mouthSmileRight);
  const frownAverage = average(scores.mouthFrownLeft, scores.mouthFrownRight);
  const browTension = average(scores.browDownLeft, scores.browDownRight);
  const eyeNarrowing = average(scores.eyeSquintLeft, scores.eyeSquintRight);
  const lowSmile = 1 - Math.max(0, Math.min(1, smileAverage));
  const synergy =
    browTension > 0.24 && eyeNarrowing > 0.18 && smileAverage < 0.24
      ? 0.22
      : 0;

  return Math.max(
    0,
    Math.min(
      1,
      browTension * 0.42 +
        eyeNarrowing * 0.28 +
        frownAverage * 0.18 +
        lowSmile * 0.1 +
        synergy,
    ),
  );
}

function getSignalRows(
  scores: CoreBlendshapeScores,
  visionMetrics: VisionMetrics | null,
): SignalRow[] {
  const quality = visionMetrics?.frameMetrics.qualityScore ?? 0;
  const faceArea = visionMetrics?.faceBounds.areaRatio ?? 0;
  const pose = visionMetrics?.headPose;
  const poseOffset = pose
    ? clamp01((Math.abs(pose.yaw) + Math.abs(pose.pitch) + Math.abs(pose.roll)) / 90)
    : 0;

  return [
    {
      label: "Angry / Tense",
      detail: "brow + eyes + frown",
      value: getAngryTensionIndex(scores),
    },
    {
      label: "Smile",
      detail: "mouth corners",
      value: average(scores.mouthSmileLeft, scores.mouthSmileRight),
    },
    {
      label: "Brow",
      detail: "downward pressure",
      value: average(scores.browDownLeft, scores.browDownRight),
    },
    {
      label: "Eyes",
      detail: "squint signal",
      value: average(scores.eyeSquintLeft, scores.eyeSquintRight),
    },
    {
      label: "Jaw",
      detail: "opening",
      value: scores.jawOpen,
    },
    {
      label: "Inner Brow",
      detail: "lift",
      value: scores.browInnerUp,
    },
    {
      label: "Frown",
      detail: "mouth downturn",
      value: average(scores.mouthFrownLeft, scores.mouthFrownRight),
    },
    {
      label: "Face Scale",
      detail: "frame coverage",
      value: clamp01(faceArea * 4),
    },
    {
      label: "Frame Quality",
      detail: "light + sharpness",
      value: quality,
    },
    {
      label: "Head Pose",
      detail: "angle offset",
      value: poseOffset,
    },
  ];
}

export default function FaceExpressionAnalyzer() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const latestScoresRef = useRef<CoreBlendshapeScores>(createEmptyScores());
  const latestAllScoresRef = useRef<AllBlendshapeScores>({});
  const latestVisionMetricsRef = useRef<VisionMetrics | null>(null);
  const scoreHistoryRef = useRef<ScoreSample[]>([]);
  const lastUiUpdateRef = useRef(0);
  const lastSampleAtRef = useRef(0);

  const [scores, setScores] = useState<CoreBlendshapeScores>(createEmptyScores);
  const [visionMetrics, setVisionMetrics] = useState<VisionMetrics | null>(null);
  const [hasFace, setHasFace] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [hasViewedReport, setHasViewedReport] = useState(false);
  const [sampleCount, setSampleCount] = useState(0);

  const signalRows = useMemo(
    () => getSignalRows(scores, visionMetrics),
    [scores, visionMetrics],
  );
  const signalStrength = useMemo(
    () => Math.max(...signalRows.map((signal) => signal.value)),
    [signalRows],
  );

  useEffect(() => {
    let cancelled = false;
    const originalConsoleError = console.error;
    const patchedConsoleError = (...args: unknown[]) => {
      if (isMediaPipeInfoLog(args)) {
        return;
      }

      originalConsoleError(...args);
    };

    console.error = patchedConsoleError;

    async function startCameraAndDetector() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("This browser does not support camera access.");
        }

        const video = videoRef.current;

        if (!video) {
          return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        video.srcObject = stream;
        await video.play();

        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        });

        if (cancelled) {
          landmarker.close();
          return;
        }

        landmarkerRef.current = landmarker;
        setIsReady(true);

        const detect = () => {
          if (cancelled) {
            return;
          }

          const currentVideo = videoRef.current;
          const currentLandmarker = landmarkerRef.current;

          if (
            currentVideo &&
            currentLandmarker &&
            currentVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
          ) {
            let detection: FaceLandmarkerResult;

            try {
              detection = currentLandmarker.detectForVideo(
                currentVideo,
                performance.now(),
              );
            } catch (detectionError) {
              setError(
                detectionError instanceof Error
                  ? detectionError.message
                  : "Face detection failed.",
              );
              animationFrameRef.current = requestAnimationFrame(detect);
              return;
            }

            const categories = detection.faceBlendshapes[0]?.categories ?? [];
            const landmarks = detection.faceLandmarks[0] ?? [];
            const detected =
              landmarks.length > 0 && categories.length > 0;
            const nextAllScores = detected ? extractAllScores(categories) : {};
            const nextScores = detected
              ? extractCoreScores(nextAllScores)
              : createEmptyScores();
            const canvas =
              frameCanvasRef.current ?? document.createElement("canvas");
            frameCanvasRef.current = canvas;
            const nextVisionMetrics = detected
              ? getVisionMetrics({
                  landmarks,
                  matrix: detection.facialTransformationMatrixes[0],
                  video: currentVideo,
                  stream: streamRef.current,
                  canvas,
                })
              : null;
            const now = performance.now();

            if (detected) {
              latestScoresRef.current = nextScores;
              latestAllScoresRef.current = nextAllScores;
              latestVisionMetricsRef.current = nextVisionMetrics;

              if (now - lastSampleAtRef.current >= SAMPLE_INTERVAL_MS) {
                scoreHistoryRef.current = [
                  ...scoreHistoryRef.current,
                  {
                    timestamp: now,
                    scores: nextScores,
                    allScores: nextAllScores,
                    visionMetrics: nextVisionMetrics,
                  },
                ]
                  .filter((sample) => now - sample.timestamp <= SAMPLE_WINDOW_MS)
                  .slice(-MAX_SCORE_SAMPLES);
                lastSampleAtRef.current = now;
              }
            }

            if (now - lastUiUpdateRef.current > 150) {
              setHasFace(detected);
              setScores(nextScores);
              setVisionMetrics(nextVisionMetrics);
              setSampleCount(detected ? scoreHistoryRef.current.length : 0);
              lastUiUpdateRef.current = now;
            }
          }

          animationFrameRef.current = requestAnimationFrame(detect);
        };

        animationFrameRef.current = requestAnimationFrame(detect);
      } catch (cameraError) {
        if (!cancelled) {
          setError(
            cameraError instanceof Error
              ? cameraError.message
              : "Unable to start camera or face detector.",
          );
        }
      }
    }

    startCameraAndDetector();

    return () => {
      cancelled = true;

      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      landmarkerRef.current?.close();
      landmarkerRef.current = null;

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (console.error === patchedConsoleError) {
        console.error = originalConsoleError;
      }
    };
  }, []);

  const analyzeExpression = useCallback(async () => {
    if (!hasFace) {
      return;
    }

    setIsAnalyzing(true);
    setError("");
    setResult(null);
    setIsReportOpen(false);
    setHasViewedReport(false);

    try {
      const response = await fetch("/api/analyze-expression", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          scores: latestScoresRef.current,
          allScores: latestAllScoresRef.current,
          visionMetrics: latestVisionMetricsRef.current,
          samples: scoreHistoryRef.current,
        }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error ?? "Analysis failed.");
      }

      setResult(data as AnalysisResult);
    } catch (analysisError) {
      setError(
        analysisError instanceof Error
          ? analysisError.message
          : "Unable to analyze expression.",
      );
    } finally {
      setIsAnalyzing(false);
    }
  }, [hasFace]);

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-3 md:h-full">
      <header className="grid shrink-0 gap-3 border-b border-cyan-100/15 pb-3 lg:grid-cols-[minmax(260px,auto)_minmax(0,1fr)] lg:items-end">
        <div className="min-w-0">
          <p className="truncate text-[11px] uppercase tracking-[0.4em] text-cyan-200/70">
            Local Vision / Expanded Signal AI
          </p>
          <h1 className="mt-1.5 truncate text-3xl font-semibold tracking-tight text-white md:text-4xl">
            Expression Signal Console
          </h1>
        </div>

        <div className="grid min-w-0 grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          <Metric label="Face" value={hasFace ? "Face detected" : "No face"} />
          <Metric label="Model" value={isReady ? "Ready" : "Loading"} />
          <Metric label="Signal" value={clampPercent(signalStrength)} />
          <Metric label="Window" value={`${sampleCount} samples`} />
          <Metric
            label="Quality"
            value={clampPercent(visionMetrics?.frameMetrics.qualityScore ?? 0)}
          />
          <Metric
            label="Pose"
            value={
              visionMetrics?.headPose.matrixAvailable
                ? `${Math.round(visionMetrics.headPose.yaw)}deg`
                : "Pending"
            }
          />
        </div>
      </header>

      <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[minmax(0,1fr)_400px] xl:grid-cols-[minmax(0,1fr)_440px] 2xl:grid-cols-[minmax(0,1fr)_500px]">
        <div className="flex min-h-[460px] flex-col border border-cyan-200/25 bg-[#050912]/95 p-3 shadow-[0_0_80px_rgba(20,184,166,0.18)] backdrop-blur lg:min-h-0 xl:p-4">
          <div className="relative min-h-[300px] flex-1 overflow-hidden border border-cyan-100/30 bg-black shadow-[inset_0_0_80px_rgba(8,145,178,0.16)] lg:min-h-0">
            <video
              ref={videoRef}
              className="h-full min-h-[300px] w-full object-cover opacity-95 lg:min-h-0"
              muted
              playsInline
              autoPlay
            />
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_30%,rgba(34,211,238,0.12),transparent_34%),linear-gradient(90deg,rgba(34,211,238,0.09)_1px,transparent_1px),linear-gradient(0deg,rgba(34,211,238,0.08)_1px,transparent_1px)] bg-[size:100%_100%,54px_54px,54px_54px]" />
            <div className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-cyan-300/16 to-transparent" />
            <div className="pointer-events-none absolute bottom-4 left-4 border border-cyan-200/30 bg-black/60 px-4 py-2 text-[11px] uppercase tracking-[0.28em] text-cyan-100/80 backdrop-blur">
              Expanded Vision Telemetry
            </div>
          </div>
        </div>

        <div className="grid min-h-0 gap-3 lg:grid-rows-[auto_minmax(0,1fr)]">
          <div className="border border-violet-200/25 bg-[#100d19]/95 p-4 shadow-[0_0_70px_rgba(139,92,246,0.16)] backdrop-blur xl:p-5">
            <p className="text-[11px] uppercase tracking-[0.32em] text-violet-200/70">
              AI Interpretation
            </p>
            <h2 className="mt-2 text-xl font-semibold tracking-tight xl:text-2xl">
              Visible Expression
            </h2>

            <div className="mt-3 grid grid-cols-[1fr_auto] gap-2">
              <button
                type="button"
                onClick={analyzeExpression}
                disabled={!hasFace || isAnalyzing}
                className="h-12 border border-cyan-100/40 bg-cyan-100 px-5 text-sm font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:border-slate-600 disabled:bg-slate-800 disabled:text-slate-500 xl:h-14 xl:px-6"
              >
                {isAnalyzing ? "Interpreting..." : "Analyze"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setIsReportOpen(true);
                  setHasViewedReport(true);
                }}
                disabled={!result || isAnalyzing}
                className={`h-12 border px-4 text-sm font-semibold transition disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-slate-900/60 disabled:text-slate-600 xl:h-14 xl:px-5 ${
                  hasViewedReport
                    ? "border-emerald-200/40 bg-emerald-200/10 text-emerald-50 shadow-[0_0_24px_rgba(16,185,129,0.12)] hover:border-emerald-100 hover:bg-emerald-200/15"
                    : "border-violet-200/35 bg-violet-200/10 text-violet-50 hover:border-violet-100 hover:bg-violet-200/20"
                }`}
              >
                {hasViewedReport ? "Viewed" : "Report"}
              </button>
            </div>

            {error ? (
              <div className="mt-3 border border-red-300/35 bg-red-950/40 p-3 text-sm leading-5 text-red-100">
                {error}
              </div>
            ) : null}

            <div className="mt-3 border border-cyan-100/25 bg-cyan-950/20 p-4 xl:mt-4 xl:p-5">
              <p className="text-[11px] uppercase tracking-[0.26em] text-cyan-100/65">
                Reading
              </p>
              <p className="mt-2 text-3xl font-semibold xl:text-4xl">
                {result ? EXPRESSION_LABELS[result.expression] : "Standby"}
              </p>
              <div className="mt-4 h-2.5 bg-white/10">
                <div
                  className="h-full bg-cyan-200"
                  style={{ width: result ? clampPercent(result.confidence) : "0%" }}
                />
              </div>
              <p className="mt-2 text-xs uppercase tracking-[0.2em] text-cyan-100/70">
                Confidence {result ? Math.round(result.confidence * 100) : 0}%
              </p>
            </div>
          </div>

          <SignalMatrix signalRows={signalRows} />
        </div>
      </div>

      {isReportOpen && result ? (
        <DetailedReport result={result} onClose={() => setIsReportOpen(false)} />
      ) : null}
    </section>
  );
}

function DetailedReport({
  result,
  onClose,
}: {
  result: AnalysisResult;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm md:justify-end md:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="expression-report-title"
    >
      <div className="flex h-[min(840px,calc(100vh-48px))] w-full max-w-3xl flex-col border border-cyan-200/25 bg-[#071017]/95 p-5 shadow-[0_0_90px_rgba(34,211,238,0.22)] md:mr-4">
        <div className="flex shrink-0 items-center justify-between gap-3">
          <p
            id="expression-report-title"
            className="text-[11px] uppercase tracking-[0.32em] text-cyan-100/70"
          >
            Detailed Report
          </p>
          <button
            type="button"
            onClick={onClose}
            className="border border-cyan-100/25 bg-cyan-100/10 px-3 py-1.5 text-xs font-semibold text-cyan-50 transition hover:border-cyan-100/50 hover:bg-cyan-100/20"
          >
            Close
          </button>
        </div>

        <div className="mt-3 min-h-0 flex-1 overflow-y-auto pr-1 text-sm leading-5 text-slate-200">
          <p className="border border-cyan-100/15 bg-cyan-950/20 p-3 text-cyan-50">
            {result.report.summary}
          </p>

          <ReportSection title="Primary Cues">
            <BulletList items={result.report.primaryCues} />
          </ReportSection>

          <ReportSection title="Counter Signals">
            <BulletList items={result.report.counterSignals} />
          </ReportSection>

          <ReportSection title="Alternatives">
            <div className="grid gap-2">
              {result.report.alternatives.map((alternative) => (
                <div
                  key={`${alternative.expression}-${alternative.reason}`}
                  className="border border-white/10 bg-white/[0.03] p-2"
                >
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="font-semibold text-cyan-50">
                      {EXPRESSION_LABELS[alternative.expression]}
                    </span>
                    <span className="font-mono text-cyan-100">
                      {formatConfidence(alternative.confidence)}
                    </span>
                  </div>
                  <p className="mt-1 text-xs leading-5 text-slate-400">
                    {alternative.reason}
                  </p>
                </div>
              ))}
            </div>
          </ReportSection>

          <ReportSection title="Temporal Notes">
            <BulletList items={result.report.temporalNotes} />
          </ReportSection>

          <ReportSection title="Signal Highlights">
            <div className="grid gap-2">
              {result.report.signalHighlights.map((signal) => (
                <div
                  key={`${signal.name}-${signal.note}`}
                  className="grid grid-cols-[1fr_auto] gap-3 border-b border-white/10 pb-2 last:border-b-0 last:pb-0"
                >
                  <div>
                    <p className="text-xs font-semibold text-slate-100">
                      {getSignalLabel(signal.name)}
                    </p>
                    <p className="text-xs leading-5 text-slate-500">
                      {signal.note}
                    </p>
                  </div>
                  <span className="font-mono text-xs text-cyan-100">
                    {signal.score.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </ReportSection>
        </div>
      </div>
    </div>
  );
}

function SignalMatrix({ signalRows }: { signalRows: SignalRow[] }) {
  return (
    <div className="flex min-h-[260px] flex-col overflow-hidden border border-amber-200/20 bg-[#15110b]/95 p-4 shadow-[0_0_50px_rgba(251,191,36,0.08)] backdrop-blur lg:min-h-0 xl:p-5">
      <p className="shrink-0 text-[11px] uppercase tracking-[0.32em] text-amber-200/70">
        Signal Matrix
      </p>
      <div className="mt-3 grid min-h-0 flex-1 gap-3 overflow-y-auto pr-1 xl:mt-4 xl:gap-4">
        {signalRows.map((signal) => (
          <div key={signal.label}>
            <div className="mb-1.5 flex items-center justify-between gap-4 text-xs">
              <span className="font-medium text-amber-50">{signal.label}</span>
              <span className="text-slate-400">{signal.detail}</span>
            </div>
            <div className="h-1.5 bg-white/10">
              <div
                className="h-full bg-amber-200"
                style={{ width: clampPercent(signal.value) }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReportSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-3">
      <p className="mb-2 text-[10px] uppercase tracking-[0.24em] text-cyan-100/55">
        {title}
      </p>
      {children}
    </div>
  );
}

function BulletList({ items }: { items: string[] }) {
  if (items.length === 0) {
    return <p className="text-xs text-slate-500">No strong counter-signal.</p>;
  }

  return (
    <ul className="grid gap-1.5 text-xs leading-5 text-slate-300">
      {items.map((item) => (
        <li key={item} className="border-l border-cyan-100/20 pl-2">
          {item}
        </li>
      ))}
    </ul>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-white/10 bg-white/[0.03] p-2.5">
      <p className="text-[10px] uppercase tracking-[0.24em] text-cyan-200/70">
        {label}
      </p>
      <p className="mt-1 truncate text-sm font-semibold text-white">{value}</p>
    </div>
  );
}
