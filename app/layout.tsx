import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://hova88.github.io/lss-explained/"),
  title: "Lift-Splat-Shoot — From Pixels to BEV",
  description: "A bilingual 17-chapter course that progressively explains every LSS input, image transform, latent-depth lift, coordinate transform, pooling step, supervision target and inference output on one real nuScenes frame.",
  openGraph:{title:"Lift-Splat-Shoot — From Pixels to BEV",description:"Mission, inputs, image, geometry, BEV, learning and proof—one real nuScenes frame, explained layer by layer.",images:[{url:"og-v3.png",width:1200,height:630}]},
  twitter:{card:"summary_large_image",title:"Lift-Splat-Shoot — From Pixels to BEV",description:"17 progressive chapters. Six cameras. 34,688 reference LiDAR points. One auditable BEV.",images:["og-v3.png"]},
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
