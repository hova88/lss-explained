import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://hova88.github.io/lss-explained/"),
  title: "Lift-Splat-Shoot Geometry Lab",
  description: "An interactive, source-audited explanation of how Lift-Splat-Shoot turns an arbitrary camera rig into a bird's-eye-view representation.",
  openGraph:{title:"Lift-Splat-Shoot Geometry Lab",description:"Six real nuScenes cameras. Every ray, depth bin, transform, splat, and BEV decision.",images:[{url:"og.png",width:1200,height:630}]},
  twitter:{card:"summary_large_image",title:"Lift-Splat-Shoot Geometry Lab",description:"Six cameras. One ego frame. Lift, splat, then shoot.",images:["og.png"]},
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
