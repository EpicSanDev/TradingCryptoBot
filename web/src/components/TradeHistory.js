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

const TradeHistory = ({ trades }) => {
  const formatPair = (pair) => {
    const match = pair.match(/^X(.+)Z(.+)$/);
    if (match) {
      return `${match[1]}/${match[2]}`;
    }
    return pair;
  };

  const sortedTrades = [...trades].sort((a, b) => 
    new Date(b.timestamp) - new Date(a.timestamp)
  );

  return (
    <Paper sx={{ p: 2, background: '#1a1f3a', height: '100%', maxHeight: '500px', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        Historique des Trades
      </Typography>

      {sortedTrades.length > 0 ? (
        <TableContainer>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Date/Heure</TableCell>
                <TableCell>Paire</TableCell>
                <TableCell align="center">Action</TableCell>
                <TableCell align="right">Volume</TableCell>
                <TableCell align="right">Prix</TableCell>
                <TableCell align="right">P&L</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedTrades.map((trade, index) => {
                const hasProfit = trade.profit_loss !== undefined;
                const isProfit = trade.profit_loss >= 0;

                return (
                  <TableRow key={index}>
                    <TableCell>
                      <Typography variant="caption">
                        {format(new Date(trade.timestamp), 'dd/MM HH:mm', { locale: fr })}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatPair(trade.pair)}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Chip 
                        label={trade.action.toUpperCase()} 
                        size="small"
                        color={trade.action === 'buy' ? 'success' : 'error'}
                      />
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2">
                        {trade.volume.toFixed(4)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2">
                        {trade.price.toFixed(2)}€
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      {hasProfit ? (
                        <Box>
                          <Typography 
                            variant="body2" 
                            color={isProfit ? 'success.main' : 'error.main'}
                            fontWeight="medium"
                          >
                            {isProfit ? '+' : ''}{trade.profit_loss.toFixed(2)}€
                          </Typography>
                          {trade.profit_loss_percent !== undefined && (
                            <Typography 
                              variant="caption" 
                              color={isProfit ? 'success.main' : 'error.main'}
                            >
                              ({isProfit ? '+' : ''}{trade.profit_loss_percent.toFixed(2)}%)
                            </Typography>
                          )}
                        </Box>
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
            Aucun trade dans l'historique
          </Typography>
        </Box>
      )}

      {sortedTrades.length > 0 && (
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="body2" color="text.secondary">
            {sortedTrades.length} trade{sortedTrades.length > 1 ? 's' : ''}
          </Typography>
          <Typography 
            variant="body2" 
            color={
              sortedTrades
                .filter(t => t.profit_loss !== undefined)
                .reduce((sum, t) => sum + t.profit_loss, 0) >= 0 
                ? 'success.main' 
                : 'error.main'
            }
            fontWeight="medium"
          >
            P&L Total: {
              sortedTrades
                .filter(t => t.profit_loss !== undefined)
                .reduce((sum, t) => sum + t.profit_loss, 0)
                .toFixed(2)
            }€
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default TradeHistory;