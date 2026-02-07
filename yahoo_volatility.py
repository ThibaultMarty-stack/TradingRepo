import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from scipy.stats import norm
from scipy.optimize import brentq

from dataclasses import dataclass
from typing import Any

@dataclass
class OptionSlice:
    underlying: str
    spot: float
    maturity_date: datetime.date
    T: float # time to maturity in years
    r: float # risk-free rate
    strikes: np.ndarray
    call_mid: np.ndarray
    put_mid: np.ndarray
    
def black_scholes_price(S0 : float, K : float, T : float, r : float, sigma : float, option_type : str) -> float:
    """
    Calculate the Black-Scholes price for a given option.
    
    Args:
        S0 (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free rate
        sigma (float): Volatility of the underlying stock
        option_type (str): "call" or "put"
    Returns:
        float: Price of the option
    """
    if option_type not in ["call", "put"]:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")
    
    if sigma <= 0 or T <= 0:
        return np.maximum(0.0, S0 - K) if option_type == "call" else np.maximum(0.0, K - S0)
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def compute_mid_price(row):
    bid = row["bid"]
    ask = row["ask"]
    last = row["lastPrice"]
    
    if bid>0 and ask>0:
        return (bid + ask) / 2
    elif last>0:
        return last
    elif bid>0:
        return bid
    elif ask>0:
        return ask
    else:
        return np.nan
    
def implied_volatility_call(S0: float, K: float, T: float, r: float, call_price: np.ndarray, sigma_min: float = 1e-6, sigma_max: float = 5.0) -> float:
    """
    Compute the implied volatility for a call option using the Black-Scholes formula and Brent's method.
    """
    # intrinsic value -> floor
    intrinsic = np.maximum(0.0, S0 - K * np.exp(-r * T))
    
    # check if the call is in the money
    if intrinsic + 1e-8 > call_price:
        return np.nan
    
    # objective function
    def f(sigma):
        return black_scholes_price(S0, K, T, r, sigma, "call") - call_price
    
    # find the root of the objective function
    try:
        iv = brentq(f, sigma_min, sigma_max, maxiter=100, xtol=1e-8)
    except ValueError:
        iv = np.nan
    
    return iv

def implied_volatility_yf(ticker: str, risk_free_asset: str = "^TNX")  -> list[OptionSlice]:
    asset = yf.Ticker(ticker)

    # Spot price
    S0 = asset.history(period="1d")["Close"].iloc[-1]
    #print(f"Spot price for {ticker}: {S0:.2f}")

    # List of expiries
    expiries = asset.options

    # Today
    today = datetime.now(timezone.utc)

    # Risk-free rate
    r = yf.Ticker(risk_free_asset).history(period="1d")["Close"].iloc[-1] / 100
    #print(f"Risk-free rate: {r:.2%}")


    # Volatility Surface

    slices = []

    for expiry_str in expiries[1:-1]:
        try :
            print(f"Processing {expiry_str}...")
            # Get expiry date in datetime and years
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            T = (expiry_date - today).days / 365
            if T <= 0.0: continue
            
            # Get the option chain
            opt_chain = asset.option_chain(expiry_str)
            calls_raw = opt_chain.calls.copy()
            puts_raw = opt_chain.puts.copy()
            
            # Compute the mid price
            calls_raw["mid"] = calls_raw.apply(compute_mid_price, axis=1)
            puts_raw["mid"] = puts_raw.apply(compute_mid_price, axis=1)
            
            calls = calls_raw[["strike", "mid"]].rename(columns={"mid": "call_mid"})
            puts = puts_raw[["strike", "mid"]].rename(columns={"mid": "put_mid"})

            # Merge the calls and puts
            merged = pd.merge(calls, puts, on="strike", how="inner")
            
            # Drop rows with missing prices
            merged = merged.dropna(subset=["call_mid", "put_mid"])

            # Drop rows with strikes below and above X percents of the spot
            threshold_min = 0.2
            threshold_max = 1 / threshold_min
            merged = merged[(merged["strike"] > threshold_min*S0) & (merged["strike"] < threshold_max*S0)]

            # Sort by strike and convert to numpy arrays
            merged = merged.sort_values(by="strike").reset_index(drop=True)
            K = merged["strike"].to_numpy()
            C_mid = merged["call_mid"].to_numpy()
            P_mid = merged["put_mid"].to_numpy()
            
            # Create the slice for the current expiry
            slice = OptionSlice(underlying=ticker, spot=S0, maturity_date=expiry_date, T=T, r=r, strikes=K, call_mid=C_mid, put_mid=P_mid)
            strikes = slice.strikes
            call_mid = slice.call_mid
            
            # Compute the implied volatility for the calls
            iv_calls = np.array([
            implied_volatility_call(slice.spot, Ki, slice.T, slice.r, Ci)
            for Ki, Ci in zip(strikes, call_mid)
            ])

            mask = ~np.isnan(iv_calls)
            print(f"Found {np.sum(mask)} valid calls for {expiry_str}.")
            slices.append({
                "expiry": expiry_str,
                "T": T,
                "strikes": strikes[mask],
                "iv_calls": iv_calls[mask],
            })
            
        except Exception as e:
            print(f"Error for expiry {expiry_str}: {e}")
            continue
    print(f"\nSuccessfully processed {len(expiries)} expiries, for a total of {np.sum([len(s['iv_calls']) for s in slices])} implied volatility points.")
    return slices