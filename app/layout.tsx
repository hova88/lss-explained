import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://hova88.github.io/lss-explained/"),
  title: "LSS Explained — Lift Uncertainty into a Shared Map",
  description: "A seven-scene, ray-first tensor walkthrough of LSS geometry, pooling and supervision, contrasted with BEVDepth on one real nuScenes frame.",
  openGraph:{title:"LSS Explained — From Pixels to BEV",description:"Depth distribution, context payload, exact coordinate transforms, Splat collisions and the LSS→BEVDepth supervision change.",images:[{url:"og.png",width:1200,height:630}]},
  twitter:{card:"summary_large_image",title:"LSS Explained — From Pixels to BEV",description:"A spatial tensor walkthrough of LSS and BEVDepth with real calibration, checkpoint evidence and reference LiDAR.",images:["og.png"]},
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
