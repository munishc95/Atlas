export type NavItem = {
  href: string;
  label: string;
};

export const navItems: NavItem[] = [
  { href: "/", label: "Dashboard" },
  { href: "/universe-data", label: "Universe & Data" },
  { href: "/strategy-lab", label: "Strategy Lab" },
  { href: "/walk-forward", label: "Walk-Forward" },
  { href: "/auto-research", label: "Auto Research" },
  { href: "/evaluations", label: "Evaluations" },
  { href: "/replay", label: "Replay" },
  { href: "/reports", label: "Reports" },
  { href: "/ops", label: "Ops" },
  { href: "/paper-trading", label: "Paper Trading" },
  { href: "/settings", label: "Settings" },
  { href: "/help", label: "Help" },
];

export const disclaimer =
  "This tool is for research and paper trading. Not financial advice. Trading involves risk. Past performance does not guarantee future results.";
