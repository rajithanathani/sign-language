import axios from 'axios';

// Task 1: Single shared API configuration instance for all frontend requests
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://sign-language-1-73a4.onrender.com';

console.log('[API Config] Initializing shared API service instance. Target Base URL:', API_BASE_URL);

const API = axios.create({
  baseURL: API_BASE_URL,
  timeout: 25000, // 25s timeout to handle Render cold starts smoothly
  headers: {
    'Content-Type': 'application/json',
  },
});

// Task 5: Log before every API request
API.interceptors.request.use(
  (config) => {
    const payloadSize = config.data ? Math.round(JSON.stringify(config.data).length / 1024) : 0;
    console.log(`[API Request Outgoing] ${config.method?.toUpperCase()} ${config.baseURL}${config.url} (Payload: ~${payloadSize} KB)`);
    return config;
  },
  (error) => {
    console.error('[API Request Exception]', error);
    return Promise.reject(error);
  }
);

// Task 5 & 9: Log responses and full error detail
API.interceptors.response.use(
  (response) => {
    console.log(`[API Response Received] Status ${response.status} from ${response.config.url}:`, response.data);
    return response;
  },
  (error) => {
    if (error.response) {
      console.error('[API Error Response]', {
        status: error.response.status,
        data: error.response.data,
        headers: error.response.headers,
        url: error.config?.url,
      });
    } else if (error.request) {
      console.error('[API Error Network/Cold Start] No response received from server:', error.message);
    } else {
      console.error('[API Error Setup]', error.message, error.stack);
    }
    return Promise.reject(error);
  }
);

export default API;
