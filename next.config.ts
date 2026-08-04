import type { NextConfig } from "next";

const basePath = process.env.GITHUB_PAGES === "true" ? "/lss-explained" : "";
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  assetPrefix: basePath,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  images: { unoptimized: true },
};

export default nextConfig;
