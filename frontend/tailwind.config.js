/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Executive Palette
                'exec-bg': '#0d1b2a',
                'exec-card': '#112233',
                'exec-border': '#1e3a5f',
                'exec-blue': '#1e90ff',
                'exec-accent': '#00d4ff',
                // Legacy Cyber colors
                'cyber-bg': '#0d1b2a',
                'cyber-card': '#112233',
                'cyber-border': '#1e3a5f',
                'cyber-cyan': '#00FFFF',
                'cyber-yellow': '#FFD700',
                'cyber-green': '#00FF88',
                'cyber-magenta': '#FF00FF',
                'cyber-gray': '#64748b',
            },
            fontFamily: {
                'sans': ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
                'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
            },
            boxShadow: {
                'neon-cyan': '0 0 15px rgba(0, 255, 255, 0.4)',
                'neon-green': '0 0 15px rgba(0, 255, 136, 0.4)',
                'neon-yellow': '0 0 15px rgba(255, 215, 0, 0.4)',
                'exec-glow': '0 0 20px rgba(30, 144, 255, 0.2)',
            },
            animation: {
                'pulse-neon': 'pulse-neon 2s ease-in-out infinite',
                'fade-in': 'fade-in 0.3s ease-out',
            },
            keyframes: {
                'pulse-neon': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.7 },
                },
                'fade-in': {
                    '0%': { opacity: 0, transform: 'translateY(4px)' },
                    '100%': { opacity: 1, transform: 'translateY(0)' },
                },
            },
        },
    },
    plugins: [],
}
