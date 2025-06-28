import axios from 'axios';
import { toast } from 'react-toastify';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Créer une instance axios avec configuration par défaut
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Intercepteur pour gérer les erreurs
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.error || 'Une erreur est survenue';
    toast.error(message);
    return Promise.reject(error);
  }
);

// Services API
export const getStatus = async () => {
  const response = await api.get('/status');
  return response.data;
};

export const getConfig = async () => {
  const response = await api.get('/config');
  return response.data;
};

export const getPositions = async () => {
  const response = await api.get('/positions');
  return response.data;
};

export const getHistory = async (limit = 50) => {
  const response = await api.get(`/history?limit=${limit}`);
  return response.data;
};

export const getPrices = async () => {
  const response = await api.get('/prices');
  return response.data;
};

export const startBot = async (mode = null) => {
  const response = await api.post('/start_bot', { mode });
  toast.success('Bot démarré avec succès');
  return response.data;
};

export const stopBot = async () => {
  const response = await api.post('/stop_bot');
  toast.success('Bot arrêté avec succès');
  return response.data;
};

export const executeTrade = async (action, pair, volume = null) => {
  const response = await api.post('/manual_trade', {
    action,
    pair,
    volume,
  });
  toast.success(`Trade ${action} exécuté avec succès`);
  return response.data;
};

export default api;