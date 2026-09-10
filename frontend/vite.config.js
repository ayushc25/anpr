import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The API the dev server proxies to. Override without editing this file:
//   VITE_API_TARGET=http://localhost:8000 npm run dev
const API_TARGET = process.env.VITE_API_TARGET || 'http://localhost:8002'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
