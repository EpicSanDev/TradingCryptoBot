import React from 'react';
import { AppBar, Toolbar, Typography, Box, Chip, IconButton } from '@mui/material';
import { AccountBalance, WifiTethering, WifiTetheringOff } from '@mui/icons-material';

const Header = ({ isConnected, botStatus }) => {
  return (
    <AppBar position="static" sx={{ background: '#1a1f3a' }}>
      <Toolbar>
        <IconButton edge="start" color="inherit" sx={{ mr: 2 }}>
          <AccountBalance />
        </IconButton>
        
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Bot de Trading Crypto Avancé
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {botStatus && (
            <Chip
              label={botStatus.is_running ? 'Bot Actif' : 'Bot Arrêté'}
              color={botStatus.is_running ? 'success' : 'error'}
              size="small"
            />
          )}
          
          {botStatus?.trading_mode && (
            <Chip
              label={botStatus.trading_mode.toUpperCase()}
              color="primary"
              size="small"
            />
          )}

          {isConnected ? (
            <WifiTethering color="success" />
          ) : (
            <WifiTetheringOff color="error" />
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;