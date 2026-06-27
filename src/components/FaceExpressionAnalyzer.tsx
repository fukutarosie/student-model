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
};

type ExpressionScoreMap = Record<AnalysisResult["expression"], number>;

type CommunicationResult = {
  summary: string;
  communicationTone:
    | "supportive"
    | "uncertain"
    | "tense"
    | "engaged"
    | "disengaged"
    | "mixed";
  confidence: number;
  visibleExpression: AnalysisResult["expression"];
  audioContentExpression: AnalysisResult["expression"];
  finalExpression: AnalysisResult["expression"];
  audioContentScores: ExpressionScoreMap;
  facialExpressionScores: ExpressionScoreMap;
  combinedExpressionScores: ExpressionScoreMap;
  spokenSignals: string[];
  facialSignals: string[];
  recommendation: string;
  warning: string;
};

type ExpressionScoreMap = Record<Expression, number>;

type CommunicationResult = {
  summary: string;
  communicationTone:
    | "supportive"
    | "uncertain"
    | "tense"
    | "engaged"
    | "disengaged"
    | "mixed";
  confidence: number;
  visibleExpression: Expression;
  audioContentExpression: Expression;
  finalExpression: Expression;
  audioContentScores: ExpressionScoreMap;
  facialExpressionScores: ExpressionScoreMap;
  combinedExpressionScores: ExpressionScoreMap;
  spokenSignals: string[];
  facialSignals: string[];
  recommendation: string;
  warning: string;
};

type SignalRow = {
  label: string;
  detail: string;
  value: number;
};

type SpeechRecognitionResultItem = {
  transcript: string;
};

type SpeechRecognitionResultListItem = {
  isFinal: boolean;
  [index: number]: SpeechRecognitionResultItem;
};

type SpeechRecognitionEvent = Event & {
  resultIndex: number;
  results: {
    length: number;
    [index: number]: SpeechRecognitionResultListItem;
  };
};

type SpeechRecognitionLike = EventTarget & {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: Event & { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
    webkitAudioContext?: typeof AudioContext;
  }
}

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
const COMMUNICATION_WARNING =
  "This combines transcript cues and visible expression estimates only. It is not proof of a real mood, intent, or mental health state.";

function isMediaPipeInfoLog(args: unknown[]) {
  return args.some(
    (arg) =>
      typeof arg === "string" &&
      arg.includes("Created TensorFlow Lite XNNPACK delegate for CPU"),
  );
}

function createEmptyScores(): BlendshapeScores {
  return Object.fromEntries(
    BLENDSHAPE_KEYS.map((key) => [key, 0]),
  ) as BlendshapeScores;
}

function extractScores(categories: Category[]): BlendshapeScores {
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
  const audioStreamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const speechRecognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const shouldListenRef = useRef(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioMonitorFrameRef = useRef<number | null>(null);
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
  const [isRecording, setIsRecording] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [recordingUrl, setRecordingUrl] = useState("");
  const [recordingMimeType, setRecordingMimeType] = useState("");
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [speechSupported, setSpeechSupported] = useState(true);
  const [speechLanguage, setSpeechLanguage] = useState("en-US");
  const [selectedAudioDeviceId, setSelectedAudioDeviceId] = useState("");
  const [audioInputDevices, setAudioInputDevices] = useState<MediaDeviceInfo[]>(
    [],
  );
  const [audioLevel, setAudioLevel] = useState(0);
  const [hasAudioSignal, setHasAudioSignal] = useState(false);
  const [microphoneStatus, setMicrophoneStatus] = useState("Idle");
  const [microphoneLabel, setMicrophoneLabel] = useState("Default microphone");
  const [communicationResult, setCommunicationResult] =
    useState<CommunicationResult | null>(null);
  const [isAnalyzingCommunication, setIsAnalyzingCommunication] =
    useState(false);
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

  const loadAudioInputDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      return;
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    setAudioInputDevices(
      devices.filter((device) => device.kind === "audioinput"),
    );
  }, []);

  const stopAudioMonitor = useCallback(() => {
    if (audioMonitorFrameRef.current !== null) {
      cancelAnimationFrame(audioMonitorFrameRef.current);
      audioMonitorFrameRef.current = null;
    }

    audioContextRef.current?.close();
    audioContextRef.current = null;
    setAudioLevel(0);
    setHasAudioSignal(false);
  }, []);

  const startAudioMonitor = useCallback(
    (stream: MediaStream) => {
      stopAudioMonitor();

      const audioTrack = stream.getAudioTracks()[0];

      if (!audioTrack) {
        setMicrophoneStatus("No microphone");
        return;
      }

      setMicrophoneStatus(audioTrack.enabled ? "Ready" : "Muted");
      setMicrophoneLabel(audioTrack.label || "Default microphone");

      try {
        const AudioContextConstructor =
          window.AudioContext ?? window.webkitAudioContext;
        const audioContext = new AudioContextConstructor();
        const source = audioContext.createMediaStreamSource(
          new MediaStream([audioTrack]),
        );
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 1024;
        const data = new Uint8Array(analyser.fftSize);
        source.connect(analyser);
        audioContextRef.current = audioContext;

        const monitor = () => {
          analyser.getByteTimeDomainData(data);

          let sum = 0;

          for (const value of data) {
            const centered = value - 128;
            sum += centered * centered;
          }

          const rms = Math.sqrt(sum / data.length) / 128;
          const nextLevel = Math.min(1, rms * 12);

          setAudioLevel(nextLevel);
          setHasAudioSignal(nextLevel > 0.01);
          audioMonitorFrameRef.current = requestAnimationFrame(monitor);
        };

        monitor();
      } catch {
        setMicrophoneStatus("Meter unavailable");
      }
    },
    [stopAudioMonitor],
  );

  const startAudioCapture = useCallback(
    async (audioDeviceId?: string) => {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not support microphone access.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: false,
        audio: audioDeviceId
          ? {
              deviceId: { exact: audioDeviceId },
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: true,
            }
          : {
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: true,
            },
      });

      audioStreamRef.current?.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = stream;
      startAudioMonitor(stream);

      const deviceId = stream.getAudioTracks()[0]?.getSettings().deviceId;

      if (deviceId) {
        setSelectedAudioDeviceId(deviceId);
      }

      await loadAudioInputDevices();

      return stream;
    },
    [loadAudioInputDevices, startAudioMonitor],
  );

  const ensureAudioStream = useCallback(async () => {
    const current = audioStreamRef.current;
    const activeTrack = current
      ?.getAudioTracks()
      .find((track) => track.readyState === "live");

    if (current && activeTrack) {
      activeTrack.enabled = true;
      setMicrophoneStatus(isListening ? "Listening" : "Ready");
      return current;
    }

    return startAudioCapture(selectedAudioDeviceId || undefined);
  }, [isListening, selectedAudioDeviceId, startAudioCapture]);

  const resumeAudioDetection = useCallback(async () => {
    try {
      if (audioContextRef.current?.state === "suspended") {
        await audioContextRef.current.resume();
      }

      await ensureAudioStream();
    } catch {
      setMicrophoneStatus("Audio resume failed");
    }
  }, [ensureAudioStream]);

  const stopSpeechRecognition = useCallback(() => {
    shouldListenRef.current = false;
    speechRecognitionRef.current?.stop();
    speechRecognitionRef.current = null;
    setIsListening(false);
    setInterimTranscript("");
    setMicrophoneStatus((current) =>
      current === "Listening" ? "Ready" : current,
    );
  }, []);

  const startSpeechRecognition = useCallback(async () => {
    if (isListening) {
      return;
    }

    try {
      await ensureAudioStream();
    } catch (microphoneError) {
      setError(
        microphoneError instanceof Error
          ? microphoneError.message
          : "Unable to start microphone.",
      );
      return;
    }

    const SpeechRecognition =
      window.SpeechRecognition ?? window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setSpeechSupported(false);
      setError(
        "Speech recognition is not available in this browser. Chrome or Edge works best.",
      );
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = speechLanguage;
    shouldListenRef.current = true;

    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";

      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const resultItem = event.results[index];
        const text = resultItem[0]?.transcript ?? "";

        if (resultItem.isFinal) {
          finalText += text;
        } else {
          interimText += text;
        }
      }

      if (finalText.trim()) {
        setTranscript((current) =>
          `${current}${current ? " " : ""}${finalText.trim()}`.trim(),
        );
      }

      setInterimTranscript(interimText.trim());
    };

    recognition.onerror = (event) => {
      if (event.error === "no-speech") {
        return;
      }

      setError(`Speech recognition issue: ${event.error ?? "unknown error"}.`);
    };

    recognition.onend = () => {
      if (shouldListenRef.current) {
        try {
          recognition.start();
        } catch {
          setIsListening(false);
        }
        return;
      }

      setIsListening(false);
    };

    speechRecognitionRef.current = recognition;

    try {
      recognition.start();
      setIsListening(true);
      setSpeechSupported(true);
      setMicrophoneStatus("Listening");
    } catch {
      setError("Speech recognition could not be started.");
      setIsListening(false);
    }
  }, [ensureAudioStream, isListening, speechLanguage]);

  const switchAudioDevice = useCallback(
    async (deviceId: string) => {
      if (!deviceId || deviceId === selectedAudioDeviceId) {
        return;
      }

      try {
        stopSpeechRecognition();
        setError("");
        setMicrophoneStatus("Switching mic");
        await startAudioCapture(deviceId);
        await audioContextRef.current?.resume();
      } catch (audioError) {
        setError(
          audioError instanceof Error
            ? audioError.message
            : "Unable to switch microphone.",
        );
        setMicrophoneStatus("Mic switch failed");
      }
    },
    [selectedAudioDeviceId, startAudioCapture, stopSpeechRecognition],
  );

  const toggleMicrophone = useCallback(() => {
    const audioTracks = audioStreamRef.current?.getAudioTracks() ?? [];

    if (audioTracks.length === 0) {
      void resumeAudioDetection();
      return;
    }

    const shouldEnable = audioTracks.some((track) => !track.enabled);

    for (const track of audioTracks) {
      track.enabled = shouldEnable;
    }

    setMicrophoneStatus(shouldEnable ? "Ready" : "Muted");

    if (!shouldEnable) {
      stopSpeechRecognition();
    }
  }, [resumeAudioDetection, stopSpeechRecognition]);

  const startRecording = useCallback(async () => {
    if (!window.MediaRecorder) {
      setError("MediaRecorder is not available in this browser.");
      return;
    }

    try {
      const audioStream = await ensureAudioStream();
      const videoTracks = streamRef.current?.getVideoTracks() ?? [];
      const audioTracks = audioStream.getAudioTracks();
      const combinedStream = new MediaStream([...videoTracks, ...audioTracks]);
      const preferredTypes = [
        "video/webm;codecs=vp9,opus",
        "video/webm;codecs=vp8,opus",
        "video/webm",
      ];
      const mimeType =
        preferredTypes.find((type) => MediaRecorder.isTypeSupported(type)) ??
        "";
      const recorder = new MediaRecorder(
        combinedStream,
        mimeType ? { mimeType } : undefined,
      );

      if (recordingUrl) {
        URL.revokeObjectURL(recordingUrl);
      }

      recordedChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, {
          type: recorder.mimeType || "video/webm",
        });
        const url = URL.createObjectURL(blob);

        setRecordingUrl(url);
        setRecordingMimeType(blob.type);
        setIsRecording(false);
      };

      mediaRecorderRef.current = recorder;
      setError("");
      setCommunicationResult(null);
      recorder.start(1000);
      setIsRecording(true);
      void startSpeechRecognition();
    } catch (recordingError) {
      setError(
        recordingError instanceof Error
          ? recordingError.message
          : "Unable to start recording.",
      );
    }
  }, [ensureAudioStream, recordingUrl, startSpeechRecognition]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }

    stopSpeechRecognition();
    setIsRecording(false);
  }, [stopSpeechRecognition]);

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

        const stream = await getMediaStream();

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        await applyMediaStream(stream);
        const audioStream = await getAudioStream();

        if (cancelled) {
          audioStream.getTracks().forEach((track) => track.stop());
          return;
        }

        await applyAudioStream(audioStream);

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

      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }

      shouldListenRef.current = false;
      speechRecognitionRef.current?.stop();
      speechRecognitionRef.current = null;
      stopAudioMonitor();

      audioStreamRef.current?.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      audioStreamRef.current?.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;

      if (console.error === patchedConsoleError) {
        console.error = originalConsoleError;
      }
    };
  }, [stopAudioMonitor]);

  useEffect(() => {
    return () => {
      if (recordingUrl) {
        URL.revokeObjectURL(recordingUrl);
      }
    };
  }, [recordingUrl]);

  const switchAudioDevice = useCallback(
    async (deviceId: string) => {
      if (!deviceId || deviceId === selectedAudioDeviceId) {
        return;
      }

      try {
        stopSpeechRecognition();
        setError("");
        setMicrophoneStatus("Switching microphone");
        await refreshAudioStream(deviceId);
        setMicrophoneStatus("Ready");
      } catch (switchError) {
        setError(
          switchError instanceof Error
            ? switchError.message
            : "Unable to switch microphone.",
        );
      }
    },
    [
      refreshAudioStream,
      selectedAudioDeviceId,
      stopSpeechRecognition,
    ],
  );

  const toggleMicrophone = useCallback(() => {
    const audioTracks = audioStreamRef.current?.getAudioTracks() ?? [];
    const shouldEnable = audioTracks.some((track) => !track.enabled);

    for (const track of audioTracks) {
      track.enabled = shouldEnable;
    }

    setMicrophoneStatus(shouldEnable ? "Ready" : "Muted");

    if (!shouldEnable) {
      stopSpeechRecognition();
    } else {
      void resumeAudioDetection();
    }
  }, [resumeAudioDetection, stopSpeechRecognition]);

  const startRecording = useCallback(async () => {
    await resumeAudioDetection();

    let stream = streamRef.current;

    if (!stream) {
      setError("Camera and microphone are not ready yet.");
      return;
    }

    if (!window.MediaRecorder) {
      setError("This browser does not support session recording.");
      return;
    }

    if (recordingUrl) {
      URL.revokeObjectURL(recordingUrl);
      setRecordingUrl("");
    }

    let videoTracks = stream
      .getVideoTracks()
      .filter((track) => track.readyState === "live");
    let audioStream = audioStreamRef.current;
    let audioTracks = (audioStream?.getAudioTracks() ?? [])
      .filter((track) => track.readyState === "live");

    if (videoTracks.length === 0 || audioTracks.length === 0) {
      try {
        if (videoTracks.length === 0) {
          const refreshedStream = await getMediaStream();
          await applyMediaStream(refreshedStream);
          stream = refreshedStream;
          videoTracks = stream
            .getVideoTracks()
            .filter((track) => track.readyState === "live");
        }

        if (audioTracks.length === 0) {
          audioStream = await refreshAudioStream(selectedAudioDeviceId);
          audioTracks = audioStream
            .getAudioTracks()
            .filter((track) => track.readyState === "live");
        }
      } catch {
        setError("Unable to refresh camera and microphone before recording.");
        return;
      }
    }

    if (videoTracks.length === 0) {
      setError("Recording cannot start because no live camera track was found.");
      return;
    }

    if (audioTracks.length === 0) {
      setError("Recording cannot start because no live microphone track was found.");
      setRecordingAudioStatus("No audio track");
      return;
    }

    for (const audioTrack of audioTracks) {
      audioTrack.enabled = true;
    }

    const recordingStream = new MediaStream([
      ...videoTracks,
      ...audioTracks,
    ]);
    const preferredTypes = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm",
    ];
    const mimeType =
      preferredTypes.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
    const recorder = new MediaRecorder(
      recordingStream,
      mimeType ? { mimeType } : undefined,
    );

    recordedChunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunksRef.current.push(event.data);
      }
    };
    recorder.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, {
        type: recorder.mimeType || "video/webm",
      });
      setRecordingMimeType(blob.type);
      setRecordingUrl(URL.createObjectURL(blob));
      setIsRecording(false);
      setRecordingAudioStatus(
        audioTracks.length > 0 ? "Audio track recorded" : "No audio track",
      );
    };
    recorder.onerror = () => {
      setError("Recording failed while capturing camera or microphone.");
      setIsRecording(false);
      setRecordingAudioStatus("Recording failed");
    };

    mediaRecorderRef.current = recorder;
    setError("");
    setCommunicationResult(null);
    setRecordingAudioStatus(
      `Recording ${audioTracks.length} audio track${audioTracks.length === 1 ? "" : "s"}`,
    );
    recorder.start(1000);
    setIsRecording(true);
    startSpeechRecognition();
  }, [
    applyMediaStream,
    getMediaStream,
    recordingUrl,
    refreshAudioStream,
    resumeAudioDetection,
    selectedAudioDeviceId,
    startSpeechRecognition,
  ]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }

    stopSpeechRecognition();
    setIsRecording(false);
  }, [stopSpeechRecognition]);

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

  const analyzeCommunication = useCallback(async () => {
    const currentTranscript = `${transcript} ${interimTranscript}`.trim();

    if (!currentTranscript) {
      setError("Listen to speech or type transcript text before analyzing communication.");
      return;
    }

    setIsAnalyzingCommunication(true);
    setError("");
    setCommunicationResult(null);

    try {
      const response = await fetch("/api/analyze-communication", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          transcript: currentTranscript,
          scores: latestScoresRef.current,
          allScores: latestAllScoresRef.current,
          visionMetrics: latestVisionMetricsRef.current,
          samples: scoreHistoryRef.current,
          expressionResult: result,
        }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error ?? "Communication analysis failed.");
      }

      setCommunicationResult(data as CommunicationResult);
    } catch (communicationError) {
      setError(
        communicationError instanceof Error
          ? communicationError.message
          : "Unable to analyze communication.",
      );
    } finally {
      setIsAnalyzingCommunication(false);
    }
  }, [interimTranscript, result, transcript]);

  const clearSession = useCallback(() => {
    if (recordingUrl) {
      URL.revokeObjectURL(recordingUrl);
    }

    setRecordingUrl("");
    setRecordingMimeType("");
    setTranscript("");
    setInterimTranscript("");
    setCommunicationResult(null);
    setError("");
  }, [recordingUrl]);

  return (
    <section className="min-h-0 flex-1 md:h-full">
      <div className="grid h-full min-h-0 gap-3 md:grid-cols-[minmax(0,1fr)_340px] lg:grid-cols-[minmax(0,1fr)_380px] xl:grid-cols-[minmax(0,1fr)_440px] 2xl:grid-cols-[minmax(0,1fr)_500px]">
        <div className="flex min-h-[460px] flex-col border border-cyan-200/25 bg-[#050912]/95 p-3 shadow-[0_0_80px_rgba(20,184,166,0.18)] backdrop-blur md:min-h-0 xl:p-4">
          <div className="mb-3 grid shrink-0 gap-2 xl:grid-cols-[minmax(230px,0.34fr)_minmax(0,1fr)] xl:items-end">
            <div className="min-w-0">
              <p className="truncate text-[10px] uppercase tracking-[0.34em] text-cyan-200/70">
                Local Vision / Expanded Signal AI
              </p>
              <h1 className="mt-1 truncate text-2xl font-semibold tracking-tight text-white lg:text-3xl 2xl:text-4xl">
                Expression Signal Console
              </h1>
            </div>

    setIsAnalyzingCommunication(true);
    setError("");
    setCommunicationResult(null);

    try {
      const response = await fetch("/api/analyze-communication", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          scores: latestScoresRef.current,
          samples: scoreHistoryRef.current,
          expressionResult: result,
          transcript: `${transcript} ${interimTranscript}`.trim(),
        }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error ?? "Communication analysis failed.");
      }

      setCommunicationResult(data as CommunicationResult);
    } catch (communicationError) {
      setError(
        communicationError instanceof Error
          ? communicationError.message
          : "Unable to analyze communication.",
      );
    } finally {
      setIsAnalyzingCommunication(false);
    }
  }, [interimTranscript, result, transcript]);

  const clearSession = useCallback(() => {
    if (recordingUrl) {
      URL.revokeObjectURL(recordingUrl);
    }

    setRecordingUrl("");
    setRecordingMimeType("");
    setRecordingAudioStatus("Not recording");
    setTranscript("");
    setInterimTranscript("");
    setCommunicationResult(null);
  }, [recordingUrl]);

  return (
    <section className="min-h-0 flex-1 md:h-[calc(100vh-112px)]">
      <div className="grid h-full min-h-0 gap-3 md:grid-cols-[minmax(0,1.6fr)_minmax(330px,0.75fr)]">
        <div className="flex min-h-[420px] flex-col overflow-hidden border border-cyan-200/20 bg-[#080b12]/90 p-3 shadow-[0_0_56px_rgba(20,184,166,0.12)] backdrop-blur md:min-h-0">
          <div className="mb-3 grid shrink-0 grid-cols-2 gap-2 xl:grid-cols-5">
            <Metric label="Face" value={hasFace ? "Face detected" : "No face"} />
            <Metric label="Model" value={isReady ? "Ready" : "Loading"} />
            <Metric label="Signal" value={clampPercent(signalStrength)} />
            <Metric
              label="Audio"
              value={hasAudioSignal ? "Signal detected" : microphoneStatus}
            />
            <Metric label="Window" value={`${sampleCount} samples`} />
          </div>

          <div className="relative min-h-[260px] flex-1 overflow-hidden border border-cyan-100/30 bg-black shadow-[inset_0_0_80px_rgba(8,145,178,0.16)] md:min-h-0">
            <video
              ref={videoRef}
              className="h-full min-h-[260px] w-full object-cover opacity-95 md:min-h-0"
              muted
              playsInline
              autoPlay
            />
            <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,rgba(34,211,238,0.08)_1px,transparent_1px),linear-gradient(0deg,rgba(34,211,238,0.07)_1px,transparent_1px)] bg-[size:42px_42px]" />
            <div className="pointer-events-none absolute inset-x-0 top-0 h-20 bg-gradient-to-b from-cyan-300/12 to-transparent" />
            <div className="pointer-events-none absolute bottom-3 left-3 border border-cyan-200/25 bg-black/55 px-3 py-2 text-[11px] uppercase tracking-[0.26em] text-cyan-100/75 backdrop-blur">
              Live Expression + Voice Session
            </div>
          </div>

          <div className="mt-3 shrink-0 overflow-y-auto border border-sky-200/20 bg-sky-950/15 p-3 md:max-h-[230px]">
            <div className="grid gap-2 sm:grid-cols-3">
              <button
                type="button"
                onClick={isRecording ? stopRecording : startRecording}
                disabled={!isReady}
                className="h-10 border border-cyan-100/40 bg-cyan-100 px-3 text-xs font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:border-slate-600 disabled:bg-slate-800 disabled:text-slate-500"
              >
                {isRecording ? "Stop Recording" : "Record Session"}
              </button>
              <button
                type="button"
                onClick={isListening ? stopSpeechRecognition : () => void startSpeechRecognition()}
                disabled={!speechSupported}
                className="h-10 border border-white/15 bg-white/[0.04] px-3 text-xs font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:text-slate-500"
              >
                {isListening ? "Stop Voice" : "Listen Voice"}
              </button>
              <button
                type="button"
                onClick={clearSession}
                className="h-10 border border-white/15 bg-black/30 px-3 text-xs font-semibold text-slate-200 transition hover:bg-white/10"
              >
                Clear Session
              </button>
            </div>

            <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(260px,auto)] xl:items-end">
              <div className="min-w-0">
                <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.22em] text-sky-100/70">
                  <span>Audio Detection</span>
                  <span>{Math.round(audioLevel * 100)}%</span>
                </div>
                <div className="mt-2 h-2 bg-white/10">
                  <div
                    className={`h-full ${
                      hasAudioSignal ? "bg-emerald-300" : "bg-sky-200"
                    }`}
                    style={{ width: clampPercent(audioLevel) }}
                  />
                </div>
                <p className="mt-2 truncate text-xs text-slate-400">
                  {microphoneLabel} / {microphoneStatus}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 xl:grid-cols-[minmax(120px,1fr)_95px_82px_68px]">
                <select
                  value={selectedAudioDeviceId}
                  onChange={(event) => void switchAudioDevice(event.target.value)}
                  className="h-9 min-w-0 border border-white/15 bg-black/40 px-2 text-xs text-slate-100"
                  aria-label="Microphone input device"
                >
                  {audioInputDevices.length === 0 ? (
                    <option value="">Default microphone</option>
                  ) : null}
                  {audioInputDevices.map((device, index) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Microphone ${index + 1}`}
                    </option>
                  ))}
                </select>
                <select
                  value={speechLanguage}
                  onChange={(event) => setSpeechLanguage(event.target.value)}
                  className="h-9 border border-white/15 bg-black/40 px-2 text-xs text-slate-100"
                  aria-label="Speech recognition language"
                >
                  <option value="en-US">EN US</option>
                  <option value="en-GB">EN UK</option>
                  <option value="zh-CN">ZH CN</option>
                  <option value="ja-JP">JA</option>
                  <option value="id-ID">ID</option>
                </select>
                <button
                  type="button"
                  onClick={toggleMicrophone}
                  className="h-9 border border-white/15 bg-white/[0.04] px-2 text-xs font-semibold text-slate-100 transition hover:bg-white/10"
                >
                  {microphoneStatus === "Muted" ? "Enable" : "Mute"}
                </button>
                <button
                  type="button"
                  onClick={() => void resumeAudioDetection()}
                  className="h-9 border border-sky-100/25 bg-sky-100/10 px-2 text-xs font-semibold text-sky-100 transition hover:bg-sky-100/20"
                >
                  Test
                </button>
              </div>
            </div>

            {recordingUrl ? (
              <div className="mt-3 border border-emerald-200/25 bg-emerald-950/20 p-2">
                <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.2em] text-emerald-100/70">
                  <span>Recorded Session</span>
                  <a
                    href={recordingUrl}
                    download={`communication-session.${recordingMimeType.includes("mp4") ? "mp4" : "webm"}`}
                    className="text-emerald-100 underline-offset-4 hover:underline"
                  >
                    Download
                  </a>
                </div>
                <video
                  className="mt-2 max-h-28 w-full bg-black"
                  src={recordingUrl}
                  controls
                />
              </div>
            ) : null}
          </div>
        </div>

        <div className="grid min-h-0 gap-3 md:grid-rows-[auto_minmax(200px,0.85fr)_minmax(0,0.9fr)]">
          <div className="border border-violet-200/25 bg-[#100d19]/95 p-4 shadow-[0_0_70px_rgba(139,92,246,0.16)] backdrop-blur xl:p-5">
            <p className="text-[11px] uppercase tracking-[0.32em] text-violet-200/70">
              AI Interpretation
            </p>
            <h2 className="mt-2 text-xl font-semibold tracking-tight xl:text-2xl">
              Visible Expression
            </h2>

            <button
              type="button"
              onClick={analyzeExpression}
              disabled={!hasFace || isAnalyzing}
              className="mt-3 h-12 w-full border border-cyan-100/40 bg-cyan-100 px-5 text-sm font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:border-slate-600 disabled:bg-slate-800 disabled:text-slate-500"
            >
              {isAnalyzing ? "Interpreting..." : "Analyze"}
            </button>

            <button
              type="button"
              onClick={analyzeCommunication}
              disabled={
                isAnalyzingCommunication ||
                (!transcript.trim() && !interimTranscript.trim())
              }
              className="mt-2 h-12 w-full border border-violet-100/35 bg-violet-200 px-5 text-sm font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:border-slate-600 disabled:bg-slate-800 disabled:text-slate-500"
            >
              {isAnalyzingCommunication
                ? "Reading Communication..."
                : "Analyze Communication"}
            </button>

            <button
              type="button"
              onClick={analyzeCommunication}
              disabled={
                isAnalyzingCommunication ||
                (!transcript.trim() && !interimTranscript.trim())
              }
              className="mt-2 h-11 w-full border border-violet-100/35 bg-violet-200 px-5 text-sm font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:border-slate-600 disabled:bg-slate-800 disabled:text-slate-500"
            >
              {isAnalyzingCommunication
                ? "Reading Communication..."
                : "Analyze Communication"}
            </button>

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

            {communicationResult ? (
              <CommunicationSummary result={communicationResult} />
            ) : null}
          </div>

          <TranscriptPanel
            interimTranscript={interimTranscript}
            setInterimTranscript={setInterimTranscript}
            setTranscript={setTranscript}
            transcript={transcript}
          />

          <SignalMatrix signalRows={signalRows} />
        </div>
      </div>

      {isReportOpen && result ? (
        <DetailedReport result={result} onClose={() => setIsReportOpen(false)} />
      ) : null}
    </section>
  );
}

function TranscriptPanel({
  interimTranscript,
  setInterimTranscript,
  setTranscript,
  transcript,
}: {
  interimTranscript: string;
  setInterimTranscript: (value: string) => void;
  setTranscript: (value: string) => void;
  transcript: string;
}) {
  return (
    <div className="flex min-h-0 flex-col overflow-hidden border border-amber-200/20 bg-[#15110b]/95 p-4 shadow-[0_0_50px_rgba(251,191,36,0.08)] backdrop-blur xl:p-5">
      <div className="flex shrink-0 items-center justify-between gap-3">
        <p className="text-[11px] uppercase tracking-[0.32em] text-amber-200/70">
          Voice Transcript
        </p>
        <span className="font-mono text-xs text-amber-100/70">
          {transcript.trim().split(/\s+/).filter(Boolean).length} words
        </span>
      </div>
      <textarea
        value={
          interimTranscript
            ? `${transcript}${transcript ? " " : ""}${interimTranscript}`
            : transcript
        }
        onChange={(event) => {
          setTranscript(event.target.value);
          setInterimTranscript("");
        }}
        placeholder="Listen to voice, record a session, or type transcript text here."
        className="mt-3 min-h-[140px] flex-1 resize-none border border-white/10 bg-black/25 p-3 text-sm leading-6 text-amber-50/90 outline-none placeholder:text-slate-500 focus:border-amber-100/35"
      />
    </div>
  );
}

function CommunicationSummary({ result }: { result: CommunicationResult }) {
  return (
    <div className="mt-3 border border-violet-200/25 bg-violet-950/20 p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] uppercase tracking-[0.24em] text-violet-100/70">
            Communication Tone
          </p>
          <p className="mt-1 text-xl font-semibold capitalize text-violet-50">
            {result.communicationTone}
          </p>
        </div>
        <span className="font-mono text-xs text-violet-100">
          {formatConfidence(result.confidence)}
        </span>
      </div>

      <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
        <MiniResult label="Audio Content" value={result.audioContentExpression} />
        <MiniResult label="Face Cues" value={result.visibleExpression} />
        <MiniResult label="Final" value={result.finalExpression} />
      </div>

      <div className="mt-3 grid gap-1.5 text-xs">
        {Object.entries(result.combinedExpressionScores).map(
          ([expression, score]) => (
            <div key={expression}>
              <div className="mb-1 flex items-center justify-between gap-3">
                <span className="text-slate-300">
                  {EXPRESSION_LABELS[expression as Expression]}
                </span>
                <span className="font-mono text-violet-100">
                  {Math.round(score * 100)}%
                </span>
              </div>
              <div className="h-1.5 bg-white/10">
                <div
                  className="h-full bg-violet-200"
                  style={{ width: clampPercent(score) }}
                />
              </div>
            </div>
          ),
        )}
      </div>

      <p className="mt-3 text-sm leading-5 text-slate-200">{result.summary}</p>
      <p className="mt-2 text-sm leading-5 text-cyan-100">
        {result.recommendation}
      </p>
      <p className="mt-2 text-xs leading-5 text-slate-500">
        {result.warning || COMMUNICATION_WARNING}
      </p>
    </div>
  );
}

function MiniResult({ label, value }: { label: string; value: Expression }) {
  return (
    <div className="border border-white/10 bg-black/20 p-2">
      <span className="block uppercase tracking-[0.18em] text-slate-500">
        {label}
      </span>
      <span className="mt-1 block font-semibold text-violet-100">
        {EXPRESSION_LABELS[value]}
      </span>
    </div>
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
                    <span className="mt-1 block font-semibold text-amber-100">
                      {EXPRESSION_LABELS[communicationResult.audioContentExpression]}
                    </span>
                  </div>
                  <div className="border border-white/10 bg-black/20 p-2">
                    <span className="block uppercase tracking-[0.18em] text-slate-500">
                      Face Cues
                    </span>
                    <span className="mt-1 block font-semibold text-cyan-100">
                      {EXPRESSION_LABELS[communicationResult.visibleExpression]}
                    </span>
                  </div>
                  <div className="border border-white/10 bg-black/20 p-2">
                    <span className="block uppercase tracking-[0.18em] text-slate-500">
                      Final Expression
                    </span>
                    <span className="mt-1 block font-semibold text-violet-100">
                      {EXPRESSION_LABELS[communicationResult.finalExpression]}
                    </span>
                  </div>
                </div>
                <div className="mt-3 grid gap-1.5 text-xs">
                  {Object.entries(communicationResult.combinedExpressionScores).map(
                    ([expression, score]) => (
                      <div key={expression}>
                        <div className="mb-1 flex items-center justify-between gap-3">
                          <span className="text-slate-300">
                            {EXPRESSION_LABELS[expression as AnalysisResult["expression"]]}
                          </span>
                          <span className="font-mono text-violet-100">
                            {Math.round(score * 100)}%
                          </span>
                        </div>
                        <div className="h-1.5 bg-white/10">
                          <div
                            className="h-full bg-violet-200"
                            style={{ width: clampPercent(score) }}
                          />
                        </div>
                      </div>
                    ),
                  )}
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-200">
                  {communicationResult.summary}
                </p>
                <p className="mt-2 text-sm leading-6 text-cyan-100">
                  {communicationResult.recommendation}
                </p>
                <p className="mt-2 text-xs leading-5 text-slate-400">
                  {communicationResult.warning || COMMUNICATION_WARNING}
                </p>
              </div>
            ) : null}
          </div>

          <div className="border border-white/10 bg-black/35 p-4 backdrop-blur">
            <p className="text-[11px] uppercase tracking-[0.32em] text-slate-400">
              Signal Matrix
            </p>
            <div className="mt-3 grid gap-2">
              {signalRows.slice(0, 4).map((signal) => (
                <div key={signal.label}>
                  <div className="mb-1.5 flex items-center justify-between gap-4 text-xs">
                    <span className="font-medium text-slate-100">
                      {signal.label}
                    </span>
                    <span className="text-slate-500">{signal.detail}</span>
                  </div>
                  <div className="h-1.5 bg-white/10">
                    <div
                      className="h-full bg-cyan-200"
                      style={{ width: clampPercent(signal.value) }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <dl className="mt-4 grid gap-2 text-xs">
              {topBlendshapes.map(({ key, score }) => (
                <div
                  key={key}
                  className="grid grid-cols-[1fr_auto] gap-4 border-b border-white/10 pb-2 last:border-b-0 last:pb-0"
                >
                  <dt>
                    <span className="block font-medium text-slate-100">
                      {BLENDSHAPE_LABELS[key]}
                    </span>
                    <span className="text-slate-500">{key}</span>
                  </dt>
                  <dd className="font-mono text-cyan-100">
                    {score.toFixed(3)}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 border border-white/10 bg-white/[0.03] px-2 py-1.5">
      <p className="truncate text-[9px] uppercase tracking-[0.2em] text-cyan-200/70">
        {label}
      </p>
      <p className="mt-0.5 truncate text-xs font-semibold text-white">{value}</p>
    </div>
  );
}


