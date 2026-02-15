import "./globals.css";

import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { Providers } from "@/components/providers";

export const metadata: Metadata = {
  title: "Atlas",
  description: "Adaptive Swing Trading Research and Paper Trading Platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const fastMode = process.env.NEXT_PUBLIC_ATLAS_FAST_MODE === "1";
  return (
    <html lang="en">
      <body className={fastMode ? "fast-mode" : undefined}>
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
