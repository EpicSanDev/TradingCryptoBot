import React, { useState, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import Header from './components/Header';
import Dashboard from './components/Dashboard';
import TradingPairs from './components/TradingPairs';
import PositionsList from './components/PositionsList';
import TradeHistory from './components/TradeHistory';
import PerformanceChart from './components/PerformanceChart';
import ControlPanel from './components/ControlPanel';

import { connectWebSocket, disconnectWebSocket } from './services/websocket';
import { getStatus, getConfig } from './services/api';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1f3a',
    },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
  },
});

function App() {
  const [botStatus, setBotStatus] = useState(null);
  const [config, setConfig] = useState(null);
  const [prices, setPrices] = useState({});
  const [positions, setPositions] = useState([]);
  const [recentTrades, setRecentTrades] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Charger la configuration
    loadConfig();
    
    // Se connecter au WebSocket
    const socket = connectWebSocket((data) => {
      if (data) {
        setBotStatus(data.status);
        setPrices(data.prices);
        setPositions(data.positions);
        setRecentTrades(data.recent_trades);
        setIsConnected(true);
      }
    });

    // Charger le statut initial
    loadStatus();

    return () => {
      disconnectWebSocket();
      setIsConnected(false);
    };
  }, []);

  const loadConfig = async () => {
    try {
      const configData = await getConfig();
      setConfig(configData);
    } catch (error) {
      console.error('Erreur lors du chargement de la configuration:', error);
    }
  };

  const loadStatus = async () => {
    try {
      const status = await getStatus();
      setBotStatus(status);
    } catch (error) {
      console.error('Erreur lors du chargement du statut:', error);
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <ToastContainer theme="dark" position="bottom-right" />
      
      <Header isConnected={isConnected} botStatus={botStatus} />
      
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          {/* Panneau de contrôle */}
          <Grid item xs={12}>
            <ControlPanel 
              botStatus={botStatus} 
              config={config}
              onStatusChange={loadStatus}
            />
          </Grid>

          {/* Dashboard principal */}
          <Grid item xs={12}>
            <Dashboard 
              botStatus={botStatus} 
              positions={positions}
              recentTrades={recentTrades}
            />
          </Grid>

          {/* Paires de trading et prix */}
          <Grid item xs={12} md={6}>
            <TradingPairs 
              pairs={config?.trading_pairs || []} 
              prices={prices}
            />
          </Grid>

          {/* Graphique de performance */}
          <Grid item xs={12} md={6}>
            <PerformanceChart trades={recentTrades} />
          </Grid>

          {/* Positions ouvertes */}
          <Grid item xs={12} md={6}>
            <PositionsList positions={positions} prices={prices} />
          </Grid>

          {/* Historique des trades */}
          <Grid item xs={12} md={6}>
            <TradeHistory trades={recentTrades} />
          </Grid>
        </Grid>
      </Container>
    </ThemeProvider>
  );
}

export default App;