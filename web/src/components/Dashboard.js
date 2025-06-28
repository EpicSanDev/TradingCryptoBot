import React from 'react';
import { Grid, Paper, Typography, Box } from '@mui/material';
import { TrendingUp, TrendingDown, AccountBalanceWallet, Assessment } from '@mui/icons-material';

const MetricCard = ({ title, value, icon, color = 'primary', suffix = '' }) => {
  return (
    <Paper 
      sx={{ 
        p: 3, 
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center',
        background: '#1a1f3a',
        borderRadius: 2,
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: -20,
          right: -20,
          opacity: 0.1,
        }}
      >
        {React.cloneElement(icon, { sx: { fontSize: 100 } })}
      </Box>
      
      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
        {title}
      </Typography>
      
      <Typography variant="h4" component="div" color={color}>
        {value}{suffix}
      </Typography>
    </Paper>
  );
};

const Dashboard = ({ botStatus, positions, recentTrades }) => {
  const performance = botStatus?.performance || {};
  
  const totalValue = positions.reduce((sum, pos) => {
    return sum + (pos.volume * pos.current_price || 0);
  }, 0);

  const winRate = performance.win_rate || 0;
  const totalPnL = performance.total_profit_loss || 0;
  const sharpeRatio = performance.sharpe_ratio || 0;

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6} md={3}>
        <MetricCard
          title="Valeur Totale"
          value={totalValue.toFixed(2)}
          icon={<AccountBalanceWallet />}
          color="primary"
          suffix="€"
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={3}>
        <MetricCard
          title="P&L Total"
          value={totalPnL.toFixed(2)}
          icon={totalPnL >= 0 ? <TrendingUp /> : <TrendingDown />}
          color={totalPnL >= 0 ? 'success.main' : 'error.main'}
          suffix="€"
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={3}>
        <MetricCard
          title="Taux de Réussite"
          value={winRate.toFixed(1)}
          icon={<Assessment />}
          color="info.main"
          suffix="%"
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={3}>
        <MetricCard
          title="Ratio de Sharpe"
          value={sharpeRatio.toFixed(2)}
          icon={<Assessment />}
          color="secondary.main"
        />
      </Grid>
      
      <Grid item xs={12}>
        <Paper sx={{ p: 2, background: '#1a1f3a' }}>
          <Typography variant="h6" gutterBottom>
            Résumé des Performances
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="text.secondary">
                Trades Totaux
              </Typography>
              <Typography variant="h6">
                {performance.total_trades || 0}
              </Typography>
            </Grid>
            
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="text.secondary">
                Trades Gagnants
              </Typography>
              <Typography variant="h6" color="success.main">
                {performance.winning_trades || 0}
              </Typography>
            </Grid>
            
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="text.secondary">
                Trades Perdants
              </Typography>
              <Typography variant="h6" color="error.main">
                {performance.losing_trades || 0}
              </Typography>
            </Grid>
            
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="text.secondary">
                Drawdown Actuel
              </Typography>
              <Typography variant="h6" color="warning.main">
                {(performance.current_drawdown || 0).toFixed(1)}%
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default Dashboard;