# Facial Communication Recognition

This project is a Next.js application that helps improve communication by combining non-verbal facial expression signals with verbal speech/transcript cues.

It does not identify a person, match them to a celebrity, or claim to know their real feelings. It only estimates visible expression patterns and communication cues from the available camera, microphone, and transcript data.

## Purpose

The goal is to help teachers, presenters, facilitators, or interviewers understand communication better by observing:

- Visible facial expression signals such as smile, brow tension, eye narrowing, jaw opening, and frown movement.
- Verbal communication signals from participant speech converted into transcript text.
- Combined interpretation of spoken words and visible emotion signals.
- Respectful recommendations for improving the conversation, such as asking clarifying questions or slowing down.

## Main Features

- Live facial expression detection using MediaPipe face blendshapes.
- AI-assisted expression interpretation through the server API.
- Camera and microphone session recording in the browser.
- Browser speech recognition for participant words.
- Transcript display with interim and final speech text.
- Combined communication analysis using verbal and non-verbal signals.
- Public API endpoints protected by a separate API access key.

## Tech Stack

- Next.js
- React
- TypeScript
- MediaPipe Tasks Vision
- OpenAI API
- Browser `MediaRecorder`
- Browser `SpeechRecognition` / `webkitSpeechRecognition`

## Environment Setup

Create or update `.env.local`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.5
FACIAL_PUBLIC_API_KEY=facial_live_5e4e3fec0d7c49c4850bd112fe498214
```

Use `OPENAI_API_KEY` only on the server. Do not send it to browsers or external clients.

The public API access key for this project is:

```text
facial_live_5e4e3fec0d7c49c4850bd112fe498214
```

You can rotate it anytime by changing `FACIAL_PUBLIC_API_KEY`. For multiple keys, use:

```env
FACIAL_API_KEYS=key_one,key_two,key_three
```

## Run Locally

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Use Chrome or Microsoft Edge for the best speech recognition support.

## Application Flow

1. The browser asks for camera and microphone permission.
2. MediaPipe reads face blendshape scores from the live camera.
3. The app samples facial expression scores over a short time window.
4. The user can record a session with video and audio.
5. Browser speech recognition converts participant speech into transcript text.
6. The app sends expression scores and transcript text to the server.
7. The server returns communication tone, summary, signals, and recommendations.

## Public API Authentication

Public endpoints accept either header format:

```http
x-api-key: facial_live_5e4e3fec0d7c49c4850bd112fe498214
```

or:

```http
Authorization: Bearer facial_live_5e4e3fec0d7c49c4850bd112fe498214
```

## Public API Endpoints

### Health Check

```http
GET /api/public/health
```

Example:

```bash
curl http://localhost:3000/api/public/health
```

### Analyze Facial Expression

```http
POST /api/public/expression
```

This endpoint accepts MediaPipe blendshape scores and optional recent samples.

Example:

```bash
curl -X POST http://localhost:3000/api/public/expression \
  -H "Content-Type: application/json" \
  -H "x-api-key: facial_live_5e4e3fec0d7c49c4850bd112fe498214" \
  -d '{
    "scores": {
      "mouthSmileLeft": 0.42,
      "mouthSmileRight": 0.39,
      "browDownLeft": 0.08,
      "browDownRight": 0.09,
      "eyeSquintLeft": 0.16,
      "eyeSquintRight": 0.14,
      "jawOpen": 0.12,
      "browInnerUp": 0.07,
      "mouthFrownLeft": 0.03,
      "mouthFrownRight": 0.04
    },
    "samples": []
  }'
```

Response shape:

```json
{
  "expression": "happy",
  "confidence": 0.78,
  "reason": "Visible score patterns explain the estimate.",
  "warning": "This is only an expression estimate, not a real mood or mental health diagnosis."
}
```

### Analyze Communication

```http
POST /api/public/communication
```

This endpoint accepts transcript text plus facial expression scores.

Example:

```bash
curl -X POST http://localhost:3000/api/public/communication \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer facial_live_5e4e3fec0d7c49c4850bd112fe498214" \
  -d '{
    "transcript": "I understand the first part, but I am confused about the next instruction.",
    "scores": {
      "mouthSmileLeft": 0.08,
      "mouthSmileRight": 0.07,
      "browDownLeft": 0.22,
      "browDownRight": 0.2,
      "eyeSquintLeft": 0.18,
      "eyeSquintRight": 0.17,
      "jawOpen": 0.1,
      "browInnerUp": 0.31,
      "mouthFrownLeft": 0.18,
      "mouthFrownRight": 0.16
    },
    "samples": [],
    "expressionResult": {
      "expression": "unclear",
      "confidence": 0.48
    }
  }'
```

Response shape:

```json
{
  "summary": "The participant appears to be asking for clarification.",
  "communicationTone": "uncertain",
  "confidence": 0.74,
  "visibleExpression": "unclear",
  "spokenSignals": ["The transcript contains confusion or clarification language."],
  "facialSignals": ["Visible brow or frown signals may suggest concentration."],
  "recommendation": "Pause and restate the next instruction in simpler steps.",
  "warning": "This combines visible expression and transcript cues only. It is not a diagnosis or proof of a participant's real feelings."
}
```

## Safety Notes

- This project should support communication, not judge people.
- Facial expressions are not reliable proof of emotion, intent, truthfulness, or personality.
- Speech transcripts may be inaccurate depending on microphone quality, accent, background noise, and browser support.
- Ask for participant consent before recording audio or video.
- Store recordings carefully and delete them when they are no longer needed.

## Project Files

- `src/components/FaceExpressionAnalyzer.tsx`: Main camera, recording, transcript, and UI logic.
- `src/app/api/analyze-expression/route.ts`: Server-side expression analysis.
- `src/app/api/analyze-communication/route.ts`: Server-side combined communication analysis.
- `src/app/api/public/expression/route.ts`: Public key-protected expression endpoint.
- `src/app/api/public/communication/route.ts`: Public key-protected communication endpoint.
- `src/app/api/public/health/route.ts`: Public API health endpoint.

