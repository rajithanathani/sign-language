import axios from 'axios';

// Production Render API URL with localhost fallback
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://sign-language-1-73a4.onrender.com';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;
