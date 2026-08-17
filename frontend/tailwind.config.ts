import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        nexus: {
          50: "#f0f6ff",
          100: "#e0edff",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          900: "#0f172a",
        },
      },
    },
  },
  plugins: [],
};
export default config;
