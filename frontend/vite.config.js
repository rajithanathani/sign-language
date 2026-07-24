import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite configuration for Sign Language Recognition Frontend
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
