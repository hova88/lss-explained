import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://hova88.github.io/lss-explained/"),
  title: "Lift-Splat-Shoot Coordinate Geometry Lab",
  description: "A bilingual 15-chapter lab tracing six real nuScenes cameras through every LSS coordinate transform, with synchronized LiDAR, 3D, BEV and checkpoint evidence.",
  openGraph:{title:"Lift-Splat-Shoot Coordinate Geometry Lab",description:"Every matrix. One real nuScenes frame. Camera, LiDAR, ego and BEV linked together.",images:[{url:"og.png",width:1200,height:630}]},
  twitter:{card:"summary_large_image",title:"Lift-Splat-Shoot Coordinate Geometry Lab",description:"15 chapters. Six cameras. 34,688 reference LiDAR points. One auditable BEV.",images:["og.png"]},
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
