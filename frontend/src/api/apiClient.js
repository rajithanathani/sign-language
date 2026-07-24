import axios from 'axios';

// Production Render API URL with fallback
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://sign-language-1-73a4.onrender.com';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request Interceptor: Log outgoing API requests
apiClient.interceptors.request.use(
  (config) => {
    const payloadSize = config.data ? Math.round(JSON.stringify(config.data).length / 1024) : 0;
    console.log(`[API Request] ${config.method.toUpperCase()} ${config.url} (${payloadSize} KB)`);
    return config;
  },
  (error) => {
    console.error('[API Request Error]', error);
    return Promise.reject(error);
  }
);

// Response Interceptor: Log incoming API responses
apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API Response] ${response.status} ${response.config.url}`, response.data);
    return response;
  },
  (error) => {
    if (error.response) {
      console.warn(`[API Error] Status ${error.response.status} ${error.config?.url}:`, error.response.data);
    } else if (error.request) {
      console.warn(`[API Network Error / Cold Start] No response from ${error.config?.url}:`, error.message);
    } else {
      console.error('[API Error]', error.message);
    }
    return Promise.reject(error);
  }
);

export default apiClient;
