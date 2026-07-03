import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/chat",
        destination: `${process.env.BACKEND_API_URL || "http://127.0.0.1:8000"}/api/chat`,
      },
    ];
  },
};

export default nextConfig;
