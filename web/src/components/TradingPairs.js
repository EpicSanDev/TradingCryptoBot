import React from 'react';
import { 
  Paper, 
  Typography, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Chip,
  Box
} from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';

const TradingPairs = ({ pairs, prices }) => {
  const getPriceChange = (currentPrice, previousPrice) => {
    if (!previousPrice || previousPrice === 0) return 0;
    return ((currentPrice - previousPrice) / previousPrice) * 100;
  };

  const formatPair = (pair) => {
    // Convertir XXBTZEUR en BTC/EUR
    const match = pair.match(/^X(.+)Z(.+)$/);
    if (match) {
      return `${match[1]}/${match[2]}`;
    }
    return pair;
  };

  return (
    <Paper sx={{ p: 2, background: '#1a1f3a', height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Paires de Trading
      </Typography>

      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Paire</TableCell>
              <TableCell align="right">Prix</TableCell>
              <TableCell align="right">Variation 24h</TableCell>
              <TableCell align="center">Statut</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {pairs.map((pair) => {
              const price = prices[pair] || 0;
              const change = getPriceChange(price, price * 0.98); // Simulation
              const isPositive = change >= 0;

              return (
                <TableRow key={pair}>
                  <TableCell>
                    <Typography variant="body2" fontWeight="medium">
                      {formatPair(pair)}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2" fontWeight="medium">
                      {price.toFixed(2)}€
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                      {isPositive ? (
                        <TrendingUp color="success" fontSize="small" />
                      ) : (
                        <TrendingDown color="error" fontSize="small" />
                      )}
                      <Typography 
                        variant="body2" 
                        color={isPositive ? 'success.main' : 'error.main'}
                        sx={{ ml: 0.5 }}
                      >
                        {isPositive ? '+' : ''}{change.toFixed(2)}%
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="center">
                    <Chip 
                      label="Actif" 
                      color="success" 
                      size="small"
                      className="pulse"
                    />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      {pairs.length === 0 && (
        <Typography 
          variant="body2" 
          color="text.secondary" 
          align="center" 
          sx={{ mt: 3 }}
        >
          Aucune paire configurée
        </Typography>
      )}
    </Paper>
  );
};

export default TradingPairs;