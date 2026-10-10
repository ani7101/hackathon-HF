"""Transparent return-threshold calibration, not an identified structural jump fit."""
from datetime import datetime, timezone
import csv
import numpy as np
import httpx
from market_jumps import JumpParameters


def fit_spy(dates, adjusted_closes, last_close, threshold=3):
    prices = np.asarray(adjusted_closes, dtype=float)
    if len(prices) < 253 or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError('Need at least 253 finite positive daily prices.')
    if not np.isfinite(last_close) or last_close <= 0 or threshold <= 0:
        raise ValueError('Last close and jump threshold must be positive.')
    returns = np.diff(np.log(prices))
    center = np.median(returns)
    robust_std = np.median(np.abs(returns-center)) / .67448975
    if robust_std <= 0:
        raise ValueError('Return variation must be positive.')
    mask = np.abs(returns-center) > threshold*robust_std
    ordinary, jumps = returns[~mask], returns[mask]
    if len(ordinary) < 30:
        raise ValueError('Too few ordinary returns to estimate diffusion.')
    vol = float(ordinary.std(ddof=1)*np.sqrt(252))
    intensity = float(mask.mean()*252)
    # Jump log sizes are estimated from residual returns, not exact intraday events.
    mean = float((jumps-center).mean()) if len(jumps) else 0.0
    std = float((jumps-center).std(ddof=1)) if len(jumps) > 1 else 0.0
    kappa = np.expm1(mean+.5*std**2)
    drift = float(returns.mean()*252 + .5*vol**2
                  + intensity*(kappa-mean))
    params = JumpParameters(drift,vol,intensity,mean,std)
    return params, dict(symbol='SPY', status='historical_threshold_fit',
        first_date=str(dates[0]),last_date=str(dates[-1]),observations=len(prices),
        last_close=float(last_close),threshold_robust_std=threshold,
        identified_extreme_days=int(mask.sum()),
        caveat='Daily extremes proxy for jumps; includes diffusion on event days. '
               'Adjusted returns proxy total returns; explicit dividends not modeled. '
               'No claim these are pure Poisson event estimates or future predictions.')


def load_spy(csv_path=None, threshold=3):
    if csv_path:
        with open(csv_path) as f:
            rows=list(csv.DictReader(f))
        rows=sorted(rows,key=lambda r:r['Date'])
        params,metadata=fit_spy([r['Date'] for r in rows],
            [float(r['Adj Close']) for r in rows],float(rows[-1]['Close']),threshold)
        metadata['source']=str(csv_path)
        return params,metadata
    url='https://query1.finance.yahoo.com/v8/finance/chart/SPY'
    response=httpx.get(url,params={'range':'5y','interval':'1d'},
                       headers={'User-Agent':'Mozilla/5.0'},timeout=30)
    response.raise_for_status()
    data=response.json()['chart']['result'][0]
    adjusted=data['indicators']['adjclose'][0]['adjclose']
    closes=data['indicators']['quote'][0]['close']
    rows=[(datetime.fromtimestamp(t,timezone.utc).date().isoformat(),a,c)
          for t,a,c in zip(data['timestamp'],adjusted,closes)
          if a is not None and c is not None]
    params,metadata=fit_spy([r[0] for r in rows],[r[1] for r in rows],rows[-1][2],threshold)
    metadata['source']=url
    return params,metadata
