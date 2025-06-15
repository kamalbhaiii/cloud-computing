// frontend/src/App.tsx
import { useEffect, useRef, useState } from "react";
import io from "socket.io-client";

const socket = io("http://192.168.137.178:30060");

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [prediction, setPrediction] = useState<string>("");

  useEffect(() => {
    if (streaming) {
      navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      });
    }
  }, [streaming]);

  useEffect(() => {
    if (!streaming) return;

    const sendFrame = () => {
      if (!videoRef.current || !canvasRef.current) return;
      const ctx = canvasRef.current.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(videoRef.current, 0, 0, 640, 640);
      const dataUrl = canvasRef.current.toDataURL("image/jpeg");
      const base64 = dataUrl.split(",")[1]; // remove "data:image/jpeg;base64,"
      socket.emit("frame", base64);
    };

    const interval = setInterval(sendFrame, 5000); // 1 frame/sec
    return () => clearInterval(interval);
  }, [streaming]);

  useEffect(() => {
    socket.on("prediction", (data) => {
      if (data?.prediction) {
        setPrediction(`${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`);
      } else {
        setPrediction("No allowed object detected.");
      }
    });

    return () => {
      socket.off("prediction");
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-6">
      <h1 className="text-2xl font-bold mb-4">Edge Inference Camera Stream (Edge ICS)</h1>
      <video ref={videoRef} autoPlay width={640} height={640} className="border" />
      <canvas ref={canvasRef} width={640} height={640} hidden />
      <div className="mt-4">
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-800"
          onClick={() => setStreaming((s) => !s)}
        >
          {streaming ? "Stop" : "Start"} Camera
        </button>
      </div>
      {prediction && <p className="mt-4 text-xl">Prediction: {prediction}</p>}
    </div>
  );
}

export default App;
