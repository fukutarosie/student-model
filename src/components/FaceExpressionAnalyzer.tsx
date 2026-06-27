"use client";

import {
  FaceLandmarker,
  FilesetResolver,
  type Category,
} from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";

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

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

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

export default function FaceExpressionAnalyzer() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const latestScoresRef = useRef<BlendshapeScores>(createEmptyScores());
  const lastUiUpdateRef = useRef(0);

  const [scores, setScores] = useState<BlendshapeScores>(createEmptyScores);
  const [hasFace, setHasFace] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    let cancelled = false;

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
            const detection = currentLandmarker.detectForVideo(
              currentVideo,
              performance.now(),
            );
            const categories = detection.faceBlendshapes[0]?.categories ?? [];
            const detected =
              detection.faceLandmarks.length > 0 && categories.length > 0;
            const nextScores = detected
              ? extractScores(categories)
              : createEmptyScores();

            if (detected) {
              latestScoresRef.current = nextScores;
            }

            const now = performance.now();

            if (now - lastUiUpdateRef.current > 150) {
              setHasFace(detected);
              setScores(nextScores);
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
    };
  }, []);

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
        body: JSON.stringify({ scores: latestScoresRef.current }),
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
    <section className="w-full max-w-5xl rounded-lg border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)]">
        <div className="space-y-4">
          <div className="overflow-hidden rounded-md bg-zinc-950">
            <video
              ref={videoRef}
              className="aspect-video w-full object-cover"
              muted
              playsInline
              autoPlay
            />
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span
              className={`rounded-full px-3 py-1 text-sm font-medium ${
                hasFace
                  ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"
                  : "bg-zinc-100 text-zinc-700 dark:bg-zinc-900 dark:text-zinc-300"
              }`}
            >
              {hasFace ? "Face detected" : "No face detected"}
            </span>
            <span className="text-sm text-zinc-500">
              {isReady ? "Detector ready" : "Starting camera and model..."}
            </span>
          </div>
          <p className="text-sm leading-6 text-zinc-600 dark:text-zinc-400">
            Only MediaPipe facial blendshape scores are sent for analysis. No
            video, screenshots, identity data, or audio are uploaded.
          </p>
        </div>

        <div className="space-y-4">
          <button
            type="button"
            onClick={analyzeExpression}
            disabled={!hasFace || isAnalyzing}
            className="w-full rounded-md bg-zinc-950 px-4 py-3 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:text-zinc-600 dark:bg-zinc-100 dark:text-zinc-950 dark:hover:bg-zinc-300 dark:disabled:bg-zinc-800 dark:disabled:text-zinc-500"
          >
            {isAnalyzing ? "Analyzing..." : "Analyze Expression"}
          </button>

          {error ? (
            <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
              {error}
            </div>
          ) : null}

          {result ? (
            <div className="rounded-md border border-zinc-200 p-4 dark:border-zinc-800">
              <h2 className="text-base font-semibold">Visible expression</h2>
              <dl className="mt-3 space-y-2 text-sm">
                <div className="flex justify-between gap-4">
                  <dt className="text-zinc-500">Expression</dt>
                  <dd className="font-medium">{result.expression}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-zinc-500">Confidence</dt>
                  <dd className="font-medium">
                    {Math.round(result.confidence * 100)}%
                  </dd>
                </div>
              </dl>
              <p className="mt-3 text-sm leading-6 text-zinc-700 dark:text-zinc-300">
                {result.reason}
              </p>
              <p className="mt-3 text-xs leading-5 text-zinc-500">
                {result.warning}
              </p>
            </div>
          ) : null}

          <div className="rounded-md border border-zinc-200 p-4 dark:border-zinc-800">
            <h2 className="text-base font-semibold">Debug scores</h2>
            <dl className="mt-3 grid gap-2 text-sm">
              {BLENDSHAPE_KEYS.map((key) => (
                <div key={key} className="grid grid-cols-[1fr_auto] gap-3">
                  <dt className="truncate text-zinc-500">{key}</dt>
                  <dd className="font-mono text-zinc-900 dark:text-zinc-100">
                    {scores[key].toFixed(3)}
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
