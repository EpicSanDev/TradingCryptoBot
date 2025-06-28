import React, { useMemo } from 'react';
import { Paper, Typography, Box } from '@mui/material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';

const PerformanceChart = ({ trades }) => {
  const chartData = useMemo(() => {
    if (!trades || trades.length === 0) return [];

    // Trier les trades par date
    const sortedTrades = [...trades]
      .filter(t => t.profit_loss !== undefined)
      .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    // Calculer le P&L cumulatif
    let cumulativePnL = 0;
    const data = sortedTrades.map(trade => {
      cumulativePnL += trade.profit_loss || 0;
      return {
        date: trade.timestamp,
        pnl: cumulativePnL,
        trade: trade.profit_loss || 0,
        action: trade.action
      };
    });

    return data;
  }, [trades]);

  const formatTooltipDate = (value) => {
    return format(new Date(value), 'dd/MM/yyyy HH:mm', { locale: fr });
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Box sx={{ background: '#1a1f3a', p: 1.5, border: '1px solid #90caf9', borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary">
            {formatTooltipDate(label)}
          </Typography>
          <Typography variant="body2" color="primary">
            P&L Cumulatif: {payload[0].value.toFixed(2)}€
          </Typography>
          {payload[1] && (
            <Typography variant="body2" color={payload[1].value >= 0 ? 'success.main' : 'error.main'}>
              Trade: {payload[1].value >= 0 ? '+' : ''}{payload[1].value.toFixed(2)}€
            </Typography>
          )}
        </Box>
      );
    }
    return null;
  };

  return (
    <Paper sx={{ p: 2, background: '#1a1f3a', height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Courbe de Performance
      </Typography>

      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#90caf9" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#90caf9" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3f5f" />
            <XAxis 
              dataKey="date" 
              tick={false}
              stroke="#90caf9"
            />
            <YAxis 
              stroke="#90caf9"
              tick={{ fill: '#90caf9', fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area 
              type="monotone" 
              dataKey="pnl" 
              stroke="#90caf9" 
              fillOpacity={1} 
              fill="url(#colorPnl)"
              strokeWidth={2}
            />
            <Line 
              type="monotone" 
              dataKey="trade" 
              stroke="#f48fb1" 
              strokeWidth={0}
              dot={{ fill: '#f48fb1', r: 3 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <Box sx={{ 
          height: 300, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center' 
        }}>
          <Typography variant="body2" color="text.secondary">
            Pas de données de performance disponibles
          </Typography>
        </Box>
      )}

      {chartData.length > 0 && (
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="body2" color="text.secondary">
            {chartData.length} trades avec P&L
          </Typography>
          <Typography 
            variant="body2" 
            color={chartData[chartData.length - 1]?.pnl >= 0 ? 'success.main' : 'error.main'}
            fontWeight="medium"
          >
            P&L Final: {chartData[chartData.length - 1]?.pnl.toFixed(2)}€
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default PerformanceChart;