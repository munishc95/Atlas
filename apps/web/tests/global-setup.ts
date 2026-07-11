import { mkdirSync } from "node:fs";

export default function globalSetup(): void {
  const root = process.env.ATLAS_E2E_ROOT;
  if (!root) {
    throw new Error("ATLAS_E2E_ROOT must be configured before Playwright starts.");
  }
  mkdirSync(root, { recursive: true });
}
