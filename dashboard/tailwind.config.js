/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0b',
        panel: '#111214',
        border: '#27272a',
        text: '#f4f4f5',
        muted: '#a1a1aa',
        primary: {
          DEFAULT: '#3b82f6', // Cyber Blue
          hover: '#2563eb',
        },
        danger: {
          DEFAULT: '#ef4444', // Neon Red
          hover: '#dc2828',
          glow: 'rgba(239, 68, 68, 0.2)'
        },
        warning: {
          DEFAULT: '#f59e0b',
          hover: '#d97706',
        },
        success: {
          DEFAULT: '#10b981',
          hover: '#059669',
        }
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['Fira Code', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        'glow-danger': '0 0 20px rgba(239, 68, 68, 0.3)',
        'glow-primary': '0 0 20px rgba(59, 130, 246, 0.3)',
        'panel': '0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3)',
      }
    },
  },
  plugins: [],
}
