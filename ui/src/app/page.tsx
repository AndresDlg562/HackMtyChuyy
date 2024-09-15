"use client"
import { useState, useEffect } from "react";

// STEPS
import UploadVideos from "./steps/UploadVideos";
import Layout from "./steps/Layout";
import Heatmaps from "./steps/Heatmaps";

// COMPONENTS
import Navbar from "../components/navbar/Navbar";
import Progressbar from "../components/progressbar/Progressbar";

export default function Home() {
  const [step, setStep] = useState(1);
  const [videos, setVideos] = useState<string[]>([]);
  const [layout, setLayout] = useState(null);

  useEffect(() => {
    console.log(videos, layout);
  }, [videos, layout]);

  return (
    <div>
      <Navbar/>
      {
        step === 1 ? (
          <UploadVideos 
            onChangeStep={(data) => {
              setVideos(data.videos);
              setStep(2);
            }}
          />
        ) : step === 2 ? (
          <Layout
            video={videos[0]}
            onChangeStep={(layout) => {
              setLayout(layout);
              setStep(3);
            }}
          />
        ) : step === 3 && (
          <Heatmaps
            video={videos[0]?.replace('hackmty.6619572a94d5a1d039dc239d940199ed.r2.cloudflarestorage.com', 'pub-f9ef82ae3ee74240886857c6bf5f4495.r2.dev').replace('6619572a94d5a1d039dc239d940199ed.r2.cloudflarestorage.com/hackmty', 'pub-f9ef82ae3ee74240886857c6bf5f4495.r2.dev')}
            onChangeStep={(data) => {}}
          />
        )
      }
      <Progressbar step={step}/>
    </div>
  );
}
