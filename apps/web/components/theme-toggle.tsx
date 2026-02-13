"use client";

import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const fromStorage = window.localStorage.getItem("atlas-theme");
    const useDark = fromStorage === "dark";
    setDark(useDark);
    document.documentElement.classList.toggle("dark", useDark);
  }, []);

  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    window.localStorage.setItem("atlas-theme", next ? "dark" : "light");
  };

  return (
    <button
      type="button"
      onClick={toggle}
      className="focus-ring rounded-full border border-border px-3 py-1.5 text-sm text-muted hover:text-ink"
      aria-label="Toggle theme"
    >
      {dark ? "Dark" : "Light"}
    </button>
  );
}
