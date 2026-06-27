import FaceExpressionAnalyzer from "@/components/FaceExpressionAnalyzer";

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-y-auto bg-[radial-gradient(circle_at_50%_-10%,rgba(125,211,252,0.16),transparent_32%),radial-gradient(circle_at_85%_18%,rgba(168,85,247,0.14),transparent_26%),linear-gradient(135deg,#02040a_0%,#0d1020_46%,#031716_100%)] px-3 py-3 text-slate-50 md:h-screen md:max-h-screen md:overflow-hidden md:px-5 md:py-4">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,rgba(148,163,184,0.09)_1px,transparent_1px),linear-gradient(0deg,rgba(148,163,184,0.08)_1px,transparent_1px)] bg-[size:82px_82px]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-cyan-200/14 to-transparent" />
      <div className="relative flex min-h-full w-full max-w-none flex-col md:h-full md:min-h-0">
        <FaceExpressionAnalyzer />
      </div>
    </main>
  );
}
