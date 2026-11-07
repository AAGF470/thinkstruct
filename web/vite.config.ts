import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Builds into ../app/static so FastAPI (and the Docker image on Raya) serves the UI
// with no Node at runtime. `npm run dev` proxies /api to a local uvicorn on :8000.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: '../app/static',
    emptyOutDir: false, // keep app/static/simple.html; `npm run clean` removes old assets
    assetsDir: 'assets',
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
