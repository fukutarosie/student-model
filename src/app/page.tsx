import FaceExpressionAnalyzer from "@/components/FaceExpressionAnalyzer";

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-y-auto bg-[linear-gradient(135deg,#05070d_0%,#111827_48%,#061716_100%)] px-4 py-4 text-slate-50 md:h-screen md:max-h-screen md:overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(0deg,rgba(148,163,184,0.08)_1px,transparent_1px)] bg-[size:72px_72px]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-cyan-200/10 to-transparent" />
      <div className="relative mx-auto flex min-h-full max-w-7xl flex-col gap-4 md:h-full md:min-h-0">
        <header className="flex shrink-0 items-end justify-between gap-4 border-b border-white/10 pb-3">
          <div>
            <p className="text-[11px] uppercase tracking-[0.4em] text-cyan-200/70">
              Local Vision / Server AI
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white md:text-4xl">
              Expression Signal Console
            </h1>
          </div>
        </header>
        <FaceExpressionAnalyzer />
      </div>
    </main>
  );
}
