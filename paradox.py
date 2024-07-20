
import numpy as np
from multiprocessing import Pool
from multiprocessing import freeze_support

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
# currentPos = np.zeros(nInst)


# def getMyPosition(prcSoFar):
#     global currentPos
#     (nins, nt) = prcSoFar.shape
#     if (nt < 2):
#         return np.zeros(nins)
#     lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
#     lNorm = np.sqrt(lastRet.dot(lastRet))
#     lastRet /= lNorm
#     rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
#     currentPos = np.array([int(x) for x in currentPos+rpos])
#     return currentPos

prev_positions = np.zeros(nInst, dtype=int)
entry_prices = np.zeros(nInst)

def getMyPosition(prcSoFar):

    global prev_positions
    global entry_prices

    max_position=10
    
    nInst, nt = prcSoFar.shape
    
    positions = np.zeros(nInst, dtype=int)
    
    # MACD parameters
    short_window = 12
    long_window = 26
    signal_window = 5

    # Maximum investment amount per instrument
    maxInvestAmt = 5000  # $100,000 per instrument

    # Stop loss amount
    stop_loss_amount = 500  # $1,000 stop loss per position
    
    for inst in range(nInst):
        prices = prcSoFar[inst, :]
        current_price = prices[-1]
        
        # Calculate MACD
        short_ema = calculate_ema(prices, short_window)
        long_ema = calculate_ema(prices, long_window)
        macd = short_ema - long_ema
        
        # Calculate signal line
        signal_line = calculate_ema(macd, signal_window)
        
        # Calculate MACD histogram
        macd_histogram = macd - signal_line

        # Calculate position size based on maxInvestAmt and current price
        position_size = int(maxInvestAmt / current_price)

        # Check for stop loss
        if prev_positions[inst] != 0:
            entry_price = entry_prices[inst]
            position_value = prev_positions[inst] * (current_price - entry_price)
            # print(current_price,"," ,entry_price)
            if position_value < -stop_loss_amount:
                # print("stop loss")
                # Close position if stop loss is hit
                positions[inst] = 0
                prev_positions[inst] = 0
                entry_prices[inst] = 0
                continue
        
        # Generate trading signals
        if macd_histogram[-1] > 0 and macd_histogram[-2] <= 0:
            # Bullish crossover
            positions[inst] = position_size
            prev_positions[inst] = position_size
            entry_prices[inst] = current_price
                
        elif macd_histogram[-1] < 0 and macd_histogram[-2] >= 0:
            # Bearish crossover
            positions[inst] = -position_size
            prev_positions[inst] = position_size
            entry_prices[inst]=current_price
            
        else:
            # No signal, maintain previous position
            positions[inst] = prev_positions[inst]
    print(positions)
    return positions

def calculate_ema(prices, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    ema = np.convolve(prices, weights, mode='full')[:len(prices)]
    ema[:window] = ema[window]
    return ema

#Zero lag EMA

# def calculate_ema(prices, window):
#     lag = (window - 1) // 2
#     ema_data = 2 * prices - np.roll(prices, lag)
#     ema_data[:lag] = prices[:lag]  # Adjust the initial values
    
#     alpha = 2 / (window + 1)
#     zlema = np.zeros_like(prices)
#     zlema[0] = prices[0]
    
#     for i in range(1, len(prices)):
#         zlema[i] = alpha * ema_data[i] + (1 - alpha) * zlema[i-1]
    
#     return zlema
    