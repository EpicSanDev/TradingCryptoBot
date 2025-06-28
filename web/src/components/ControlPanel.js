import React, { useState } from 'react';
import { 
  Paper, 
  Box, 
  Button, 
  Typography, 
  ToggleButton, 
  ToggleButtonGroup,
  CircularProgress 
} from '@mui/material';
import { PlayArrow, Stop, Refresh } from '@mui/icons-material';
import { startBot, stopBot } from '../services/api';

const ControlPanel = ({ botStatus, config, onStatusChange }) => {
  const [loading, setLoading] = useState(false);
  const [tradingMode, setTradingMode] = useState(config?.trading_mode || 'spot');

  const handleStart = async () => {
    setLoading(true);
    try {
      await startBot(tradingMode);
      if (onStatusChange) {
        setTimeout(onStatusChange, 1000); // Attendre un peu avant de rafraîchir
      }
    } catch (error) {
      console.error('Erreur lors du démarrage:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await stopBot();
      if (onStatusChange) {
        setTimeout(onStatusChange, 1000);
      }
    } catch (error) {
      console.error('Erreur lors de l\'arrêt:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleModeChange = (event, newMode) => {
    if (newMode !== null) {
      setTradingMode(newMode);
    }
  };

  const isRunning = botStatus?.is_running || false;

  return (
    <Paper sx={{ p: 3, background: '#1a1f3a' }}>
      <Typography variant="h6" gutterBottom>
        Panneau de Contrôle
      </Typography>

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, flexWrap: 'wrap' }}>
        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Mode de Trading
          </Typography>
          <ToggleButtonGroup
            value={tradingMode}
            exclusive
            onChange={handleModeChange}
            disabled={isRunning}
          >
            <ToggleButton value="spot">
              Spot
            </ToggleButton>
            <ToggleButton value="futures">
              Futures
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          {!isRunning ? (
            <Button
              variant="contained"
              color="success"
              startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
              onClick={handleStart}
              disabled={loading}
              size="large"
            >
              Démarrer le Bot
            </Button>
          ) : (
            <Button
              variant="contained"
              color="error"
              startIcon={loading ? <CircularProgress size={20} /> : <Stop />}
              onClick={handleStop}
              disabled={loading}
              size="large"
            >
              Arrêter le Bot
            </Button>
          )}

          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={onStatusChange}
            disabled={loading}
          >
            Rafraîchir
          </Button>
        </Box>

        {config && (
          <Box sx={{ ml: 'auto' }}>
            <Typography variant="body2" color="text.secondary">
              Capital: {config.investment_amount}€
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {config.trading_pairs?.length || 0} paires surveillées
            </Typography>
          </Box>
        )}
      </Box>

      {botStatus?.last_check && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
          Dernière vérification: {new Date(botStatus.last_check).toLocaleString()}
        </Typography>
      )}
    </Paper>
  );
};

export default ControlPanel;