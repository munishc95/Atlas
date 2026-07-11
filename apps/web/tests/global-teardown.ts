import { rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export default function globalTeardown(): void {
  const root = process.env.ATLAS_E2E_ROOT;
  if (!root || process.env.ATLAS_E2E_EPHEMERAL_ROOT !== "1") {
    return;
  }

  const resolved = path.resolve(root);
  const testDir = path.dirname(fileURLToPath(import.meta.url));
  const safeParent = `${path.resolve(testDir, "../.atlas")}${path.sep}`;
  if (!resolved.startsWith(safeParent)) {
    throw new Error(`Refusing to remove unexpected E2E root: ${resolved}`);
  }
  rmSync(resolved, { recursive: true, force: true });
}
