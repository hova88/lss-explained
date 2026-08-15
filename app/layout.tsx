import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://hova88.github.io/lss-explained/"),
  title: "LSS Explained — Lift Uncertainty into a Shared Map",
  description: "A tensor-by-tensor spatial lecture on the original Lift-Splat-Shoot camera-to-BEV model, grounded in one real nuScenes frame.",
  openGraph:{title:"LSS Explained — From Pixels to BEV",description:"Six cameras, latent depth, explicit geometry and every tensor operation on the path to one auditable BEV.",images:[{url:"og.png",width:1200,height:630}]},
  twitter:{card:"summary_large_image",title:"LSS Explained — From Pixels to BEV",description:"A hand-drawn tensor walkthrough with real checkpoint evidence and 34,688 reference LiDAR points.",images:["og.png"]},
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
