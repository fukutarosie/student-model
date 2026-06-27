"use client";

import {
  FaceLandmarker,
  FilesetResolver,
  type Category,
  type FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const BLENDSHAPE_KEYS = [
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

type BlendshapeKey = (typeof BLENDSHAPE_KEYS)[number];
type BlendshapeScores = Record<BlendshapeKey, number>;
type ScoreSample = {
  timestamp: number;
  scores: BlendshapeScores;
};

type AnalysisResult = {
  expression:
    | "happy"
    | "neutral"
    | "sad"
    | "angry"
    | "surprised"
    | "tired"
    | "unclear";
  confidence: number;
  reason: string;
  warning: string;
};

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
  }
}

const EXPRESSION_LABELS: Record<AnalysisResult["expression"], string> = {
  happy: "Happy",
  neutral: "Neutral",
  sad: "Sad",
  angry: "Angry / Tense",
  surprised: "Surprised",
  tired: "Tired",
  unclear: "Unclear",
};

const BLENDSHAPE_LABELS: Record<BlendshapeKey, string> = {
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
const COMMUNICATION_WARNING =
  "This combines visible expression and transcript cues only. It is not a diagnosis or proof of a participant's real feelings.";

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

  for (const category of categories) {
    if (BLENDSHAPE_KEYS.includes(category.categoryName as BlendshapeKey)) {
      scores[category.categoryName as BlendshapeKey] = category.score;
    }
  }

  return scores;
}

function average(left: number, right: number) {
  return (left + right) / 2;
}

function clampPercent(value: number) {
  return `${Math.max(0, Math.min(100, Math.round(value * 100)))}%`;
}

function getAngryTensionIndex(scores: BlendshapeScores) {
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

function getSignalRows(scores: BlendshapeScores): SignalRow[] {
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
  ];
}

function getTopBlendshapes(scores: BlendshapeScores) {
  return [...BLENDSHAPE_KEYS]
    .map((key) => ({ key, score: scores[key] }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
}

export default function FaceExpressionAnalyzer() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const speechRecognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const shouldListenRef = useRef(false);
  const latestScoresRef = useRef<BlendshapeScores>(createEmptyScores());
  const scoreHistoryRef = useRef<ScoreSample[]>([]);
  const lastUiUpdateRef = useRef(0);
  const lastSampleAtRef = useRef(0);

  const [scores, setScores] = useState<BlendshapeScores>(createEmptyScores);
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
  const [communicationResult, setCommunicationResult] =
    useState<CommunicationResult | null>(null);
  const [isAnalyzingCommunication, setIsAnalyzingCommunication] =
    useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [sampleCount, setSampleCount] = useState(0);

  const signalRows = useMemo(() => getSignalRows(scores), [scores]);
  const topBlendshapes = useMemo(() => getTopBlendshapes(scores), [scores]);
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
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
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
            const detected =
              detection.faceLandmarks.length > 0 && categories.length > 0;
            const nextScores = detected
              ? extractScores(categories)
              : createEmptyScores();
            const now = performance.now();

            if (detected) {
              latestScoresRef.current = nextScores;

              if (now - lastSampleAtRef.current >= SAMPLE_INTERVAL_MS) {
                scoreHistoryRef.current = [
                  ...scoreHistoryRef.current,
                  { timestamp: now, scores: nextScores },
                ]
                  .filter((sample) => now - sample.timestamp <= SAMPLE_WINDOW_MS)
                  .slice(-MAX_SCORE_SAMPLES);
                lastSampleAtRef.current = now;
              }
            }

            if (now - lastUiUpdateRef.current > 150) {
              setHasFace(detected);
              setScores(nextScores);
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
      speechRecognitionRef.current?.stop();
      speechRecognitionRef.current = null;

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (console.error === patchedConsoleError) {
        console.error = originalConsoleError;
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      if (recordingUrl) {
        URL.revokeObjectURL(recordingUrl);
      }
    };
  }, [recordingUrl]);

  const startSpeechRecognition = useCallback(() => {
    if (isListening) {
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
    recognition.lang = navigator.language || "en-US";
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
    } catch {
      setError("Speech recognition could not be started.");
      setIsListening(false);
    }
  }, [isListening]);

  const stopSpeechRecognition = useCallback(() => {
    shouldListenRef.current = false;
    speechRecognitionRef.current?.stop();
    setIsListening(false);
    setInterimTranscript("");
  }, []);

  const startRecording = useCallback(() => {
    const stream = streamRef.current;

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

    const preferredTypes = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm",
    ];
    const mimeType =
      preferredTypes.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

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
    };

    mediaRecorderRef.current = recorder;
    setError("");
    setCommunicationResult(null);
    recorder.start(1000);
    setIsRecording(true);
    startSpeechRecognition();
  }, [recordingUrl, startSpeechRecognition]);

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

    try {
      const response = await fetch("/api/analyze-expression", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          scores: latestScoresRef.current,
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
    if (!transcript.trim() && !interimTranscript.trim()) {
      setError("Record or listen to participant words before analyzing communication.");
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
    setTranscript("");
    setInterimTranscript("");
    setCommunicationResult(null);
  }, [recordingUrl]);

  return (
    <section className="min-h-0 flex-1 md:h-[calc(100vh-112px)]">
      <div className="grid h-full min-h-0 gap-3 md:grid-cols-[minmax(0,1.6fr)_minmax(330px,0.75fr)]">
        <div className="flex min-h-[420px] flex-col border border-cyan-200/20 bg-[#080b12]/90 p-3 shadow-[0_0_56px_rgba(20,184,166,0.12)] backdrop-blur md:min-h-0">
          <div className="mb-3 grid shrink-0 grid-cols-2 gap-2 xl:grid-cols-4">
            <Metric label="Face" value={hasFace ? "Face detected" : "No face"} />
            <Metric label="Model" value={isReady ? "Ready" : "Loading"} />
            <Metric label="Signal" value={clampPercent(signalStrength)} />
            <Metric
              label="Window"
              value={`${sampleCount} samples`}
            />
          </div>

          <div className="relative min-h-[300px] flex-1 overflow-hidden border border-cyan-100/25 bg-black">
            <video
              ref={videoRef}
              className="h-full min-h-[300px] w-full object-cover opacity-90"
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

          <div className="mt-3 grid shrink-0 gap-2 sm:grid-cols-3">
            <button
              type="button"
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!isReady}
              className="h-11 border border-cyan-100/40 bg-cyan-100 px-4 text-sm font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:border-slate-600 disabled:bg-slate-800 disabled:text-slate-500"
            >
              {isRecording ? "Stop Recording" : "Record Session"}
            </button>
            <button
              type="button"
              onClick={isListening ? stopSpeechRecognition : startSpeechRecognition}
              disabled={!speechSupported}
              className="h-11 border border-white/15 bg-white/[0.04] px-4 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:text-slate-500"
            >
              {isListening ? "Stop Voice" : "Listen Voice"}
            </button>
            <button
              type="button"
              onClick={clearSession}
              className="h-11 border border-white/15 bg-black/30 px-4 text-sm font-semibold text-slate-200 transition hover:bg-white/10"
            >
              Clear Session
            </button>
          </div>

          {recordingUrl ? (
            <div className="mt-3 border border-emerald-200/25 bg-emerald-950/20 p-3">
              <div className="flex items-center justify-between gap-3 text-xs uppercase tracking-[0.22em] text-emerald-100/70">
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
                className="mt-2 max-h-40 w-full bg-black"
                src={recordingUrl}
                controls
              />
            </div>
          ) : null}
        </div>

        <div className="grid min-h-0 gap-3 md:grid-rows-[auto_1fr_auto]">
          <div className="border border-violet-200/20 bg-[#11101a]/90 p-4 shadow-[0_0_56px_rgba(139,92,246,0.12)] backdrop-blur">
            <p className="text-[11px] uppercase tracking-[0.32em] text-violet-200/70">
              AI Interpretation
            </p>
            <h2 className="mt-2 text-xl font-semibold tracking-tight">
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

            {error ? (
              <div className="mt-3 border border-red-300/35 bg-red-950/40 p-3 text-sm leading-5 text-red-100">
                {error}
              </div>
            ) : null}

            <div className="mt-3 border border-cyan-100/25 bg-cyan-950/20 p-4">
              <p className="text-[11px] uppercase tracking-[0.26em] text-cyan-100/65">
                Reading
              </p>
              <p className="mt-2 text-3xl font-semibold">
                {result ? EXPRESSION_LABELS[result.expression] : "Standby"}
              </p>
              <div className="mt-3 h-2 bg-white/10">
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

          <div className="min-h-0 overflow-y-auto border border-amber-200/20 bg-[#15110b]/90 p-4 backdrop-blur">
            <p className="text-[11px] uppercase tracking-[0.32em] text-amber-200/70">
              Voice Transcript
            </p>
            <div className="mt-3 min-h-28 border border-white/10 bg-black/25 p-3 text-sm leading-6 text-amber-50/90">
              {transcript || interimTranscript ? (
                <>
                  {transcript}
                  {interimTranscript ? (
                    <span className="text-amber-100/55"> {interimTranscript}</span>
                  ) : null}
                </>
              ) : (
                <span className="text-slate-500">
                  Start recording or voice listening to capture participant words.
                </span>
              )}
            </div>

            {communicationResult ? (
              <div className="mt-3 border border-violet-200/25 bg-violet-950/20 p-3">
                <p className="text-[11px] uppercase tracking-[0.24em] text-violet-100/70">
                  Communication Tone
                </p>
                <p className="mt-1 text-2xl font-semibold capitalize">
                  {communicationResult.communicationTone}
                </p>
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
    <div className="border border-white/10 bg-white/[0.03] p-2.5">
      <p className="text-[10px] uppercase tracking-[0.24em] text-cyan-200/70">
        {label}
      </p>
      <p className="mt-1 truncate text-sm font-semibold text-white">{value}</p>
    </div>
  );
}
