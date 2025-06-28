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
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';

const PositionsList = ({ positions, prices }) => {
  const formatPair = (pair) => {
    const match = pair.match(/^X(.+)Z(.+)$/);
    if (match) {
      return `${match[1]}/${match[2]}`;
    }
    return pair;
  };

  return (
    <Paper sx={{ p: 2, background: '#1a1f3a', height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Positions Ouvertes
      </Typography>

      {positions.length > 0 ? (
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Paire</TableCell>
                <TableCell align="right">Volume</TableCell>
                <TableCell align="right">Prix d'entrée</TableCell>
                <TableCell align="right">Prix actuel</TableCell>
                <TableCell align="right">P&L</TableCell>
                <TableCell align="center">Levier</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {positions.map((position, index) => {
                const currentPrice = position.current_price || prices[position.pair] || 0;
                const pnl = position.unrealized_pnl || 0;
                const pnlPercent = position.unrealized_pnl_percent || 0;
                const isProfit = pnl >= 0;

                return (
                  <TableRow key={index}>
                    <TableCell>
                      <Box>
                        <Typography variant="body2" fontWeight="medium">
                          {formatPair(position.pair)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {position.entry_time && format(new Date(position.entry_time), 'dd/MM HH:mm', { locale: fr })}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2">
                        {position.volume.toFixed(4)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2">
                        {position.entry_price.toFixed(2)}€
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2">
                        {currentPrice.toFixed(2)}€
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Box>
                        <Typography 
                          variant="body2" 
                          color={isProfit ? 'success.main' : 'error.main'}
                          fontWeight="medium"
                        >
                          {isProfit ? '+' : ''}{pnl.toFixed(2)}€
                        </Typography>
                        <Typography 
                          variant="caption" 
                          color={isProfit ? 'success.main' : 'error.main'}
                        >
                          ({isProfit ? '+' : ''}{pnlPercent.toFixed(2)}%)
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="center">
                      {position.leverage ? (
                        <Chip 
                          label={`${position.leverage}x`} 
                          size="small" 
                          color="primary"
                        />
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          -
                        </Typography>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      ) : (
        <Box sx={{ textAlign: 'center', mt: 4 }}>
          <Typography variant="body2" color="text.secondary">
            Aucune position ouverte
          </Typography>
        </Box>
      )}

      {positions.length > 0 && (
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="body2" color="text.secondary">
            Total: {positions.length} position{positions.length > 1 ? 's' : ''}
          </Typography>
          <Typography 
            variant="body2" 
            color={
              positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0) >= 0 
                ? 'success.main' 
                : 'error.main'
            }
            fontWeight="medium"
          >
            P&L Total: {positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0).toFixed(2)}€
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default PositionsList;