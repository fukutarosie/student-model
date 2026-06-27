import FaceExpressionAnalyzer from "@/components/FaceExpressionAnalyzer";

export default function Home() {
  return (
    <main className="min-h-screen bg-zinc-50 px-6 py-10 text-zinc-950 dark:bg-black dark:text-zinc-50">
      <div className="mx-auto flex max-w-5xl flex-col gap-8">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">
            摄像头人脸表情倾向分析
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-600 dark:text-zinc-400">
            在浏览器本地使用 MediaPipe 提取可见脸部表情数值，再发送这些
            blendshape scores 到服务端生成简短分析。该功能不读取真实心情，
            不做心理诊断，也不识别身份。
          </p>
        </div>
        <FaceExpressionAnalyzer />
      </div>
    </main>
  );
}
