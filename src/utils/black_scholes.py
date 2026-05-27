"""
Junior Aladdin - Black-Scholes Module
======================================

PURPOSE:
    Compute option prices, Implied Volatility (IV), and Greeks
    (Delta, Gamma, Theta, Vega) for NIFTY options.

    Angel One does NOT provide IV or Greeks — we compute them
    locally using the Black-Scholes-Merton model.

USAGE:
    from src.utils.black_scholes import (
        black_scholes_price, implied_volatility,
        compute_delta, compute_gamma, compute_theta, compute_vega,
        compute_all_greeks,
    )

    S = 24500       # Spot price
    K = 24500       # Strike price
    T = 3/365       # 3 days to expiry (in years)
    r = 0.065       # Risk-free rate (6.5% for India)
    sigma = 0.15    # 15% volatility

    price = black_scholes_price(S, K, T, r, sigma, "CE")
    iv = implied_volatility(100.0, S, K, T, r, "CE")
    greeks = compute_all_greeks(S, K, T, r, iv, "CE")

CONNECTS TO:
    - Option Chain Poller: computes IV for every option every 30 seconds
    - Feature Engine: uses IV rank, IV skew, Greeks for scoring
    - Risk Engine: monitors net Delta/Gamma of portfolio
    - GEX Proxy: uses Gamma x OI for market maker exposure estimate
"""

import math
from typing import Dict, Optional

from src.utils.config_loader import Config
from src.utils.logger import setup_logger


_LOG = setup_logger("black_scholes")


def _emit_log(level: str, msg: str, **fields) -> None:
    """Safe logger shim; never raises."""
    try:
        fn = getattr(_LOG, level, None)
        if fn is None:
            return
        try:
            fn(msg, **fields)
        except TypeError:
            # stdlib logger fallback
            if fields:
                fn(f"{msg} | " + ", ".join(f"{k}={v!r}" for k, v in sorted(fields.items())))
            else:
                fn(msg)
    except Exception:
        return


# ==============================================
# Standard Normal Distribution Functions
# ==============================================

def _norm_pdf(x: float) -> float:
    """
    Standard normal probability density function (PDF).
    The bell curve value at point x.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function (CDF).

    HARDENED FOR TAILS:
    - Uses erfc form for better precision in tails.
    - For |x| > 6, uses an asymptotic tail approximation.
    - Clamps output to (eps, 1-eps) to avoid returning exactly 0.0 or 1.0
      (prevents silent delta saturation artifacts).
    """
    # eps chosen above machine epsilon to guarantee 1-eps != 1.0 in float64
    eps = 1e-15

    if x == 0.0:
        return 0.5

    ax = abs(x)
    # Asymptotic tail approximation for very large |x|
    if ax > 6.0:
        # Q(x) ~ phi(x)/x * (1 - 1/x^2 + 3/x^4 - 15/x^6)
        inv = 1.0 / ax
        inv2 = inv * inv
        poly = 1.0 - inv2 + 3.0 * inv2 * inv2 - 15.0 * inv2 * inv2 * inv2
        tail = _norm_pdf(ax) * inv * poly
        # ensure tail non-negative
        if tail < 0.0:
            tail = 0.0
        cdf = (1.0 - tail) if x > 0 else tail
    else:
        # More stable tail computation than 0.5*(1+erf(.))
        # CDF(x) = 1 - 0.5*erfc(x/sqrt(2)) for x>=0
        #        = 0.5*erfc(-x/sqrt(2)) for x<0
        z = x / math.sqrt(2.0)
        if x > 0:
            cdf = 1.0 - 0.5 * math.erfc(z)
        else:
            cdf = 0.5 * math.erfc(-z)

    # Clamp away from exact 0/1
    if cdf <= eps:
        return eps
    if cdf >= 1.0 - eps:
        return 1.0 - eps
    return cdf


# ==============================================
# Black-Scholes d1, d2 Calculations
# ==============================================

def _compute_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute d1 parameter for Black-Scholes formula.

    Args:
        S: Spot price (e.g., 24500)
        K: Strike price (e.g., 24500)
        T: Time to expiry in years (e.g., 3/365)
        r: Risk-free rate (e.g., 0.065)
        sigma: Volatility (e.g., 0.15 for 15%)

    Returns:
        float: d1 value
    """
    if S <= 0 or K <= 0:
        return 0.0
    if sigma <= 0 or T <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _compute_d2(d1: float, sigma: float, T: float) -> float:
    """
    Compute d2 parameter: d2 = d1 - sigma * sqrt(T)
    """
    if T <= 0:
        return d1
    return d1 - sigma * math.sqrt(T)


def _raw_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute RAW Vega — the mathematical derivative of BS price
    with respect to sigma. This is used internally by Newton-Raphson.

    This is NOT divided by 100. It gives the change in option price
    for a change of 1.0 in sigma (i.e., 100 percentage points).

    The public compute_vega() divides by 100 to give per-1%-change.

    Args:
        S, K, T, r, sigma: Standard BS parameters

    Returns:
        float: Raw vega (dPrice/dSigma)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _compute_d1(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)


def _validate_option_type(option_type: str) -> str:
    """Strict validation: option_type must be 'CE' or 'PE'."""
    ot = str(option_type).upper()
    if ot not in ("CE", "PE"):
        raise ValueError(f"Invalid option_type: {option_type!r}. Must be 'CE' or 'PE'.")
    return ot


# ==============================================
# Black-Scholes Option Pricing
# ==============================================

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "CE",
) -> float:
    """
    Calculate theoretical option price using Black-Scholes-Merton model.

    Args:
        S: Spot price (e.g., 24500)
        K: Strike price (e.g., 24500)
        T: Time to expiry in years (e.g., 3/365 = 0.00822)
        r: Risk-free rate (e.g., 0.065 for 6.5% in India)
        sigma: Volatility as decimal (e.g., 0.15 for 15%)
        option_type: "CE" for Call, "PE" for Put

    Returns:
        float: Theoretical option price in same units as S and K

    Notes:
        - If T <= 0 (expired), returns intrinsic value
        - If sigma <= 0, returns intrinsic value
    """
    ot = _validate_option_type(option_type)

    # Handle edge cases
    if T <= 0:
        if ot == "CE":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    if sigma <= 0:
        if ot == "CE":
            return max(0.0, S - K * math.exp(-r * T))
        else:
            return max(0.0, K * math.exp(-r * T) - S)

    d1 = _compute_d1(S, K, T, r, sigma)
    d2 = _compute_d2(d1, sigma, T)

    if ot == "CE":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return max(0.0, price)


# ==============================================
# Implied Volatility (Newton-Raphson)
# ==============================================

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "CE",
    max_iterations: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> float:
    """
    Calculate Implied Volatility using Newton-Raphson iteration.

    Returns:
        float: Implied volatility as decimal (e.g., 0.15 for 15%)
        Returns 0.01 (1%) if computation fails.

    HARDENED:
    - Strict option_type validation (ValueError on invalid)
    - Safe casting of config values to int/float
    - Intrinsic lower bound uses *undiscounted* intrinsic:
        CE: max(0, S-K), PE: max(0, K-S)
      If market_price < intrinsic - tolerance => return 0.01
    - If vega is tiny AND price diff is not within tolerance: return 0.0 (explicit failure sentinel)
    - Stagnation break triggers only when diff is within tolerance AND sigma change tiny
    """
    ot = _validate_option_type(option_type)

    # Load defaults from config if not provided (safe casting)
    if max_iterations is None:
        try:
            max_iterations = int(Config.get("features", "iv_max_iterations", default=50))
        except Exception:
            max_iterations = 50
    else:
        try:
            max_iterations = int(max_iterations)
        except Exception:
            max_iterations = 50

    if tolerance is None:
        try:
            tolerance = float(Config.get("features", "iv_tolerance", default=0.10))
        except Exception:
            tolerance = 0.10
    else:
        try:
            tolerance = float(tolerance)
        except Exception:
            tolerance = 0.10

    if max_iterations <= 0:
        max_iterations = 50
    if tolerance <= 0:
        tolerance = 0.10

    # Edge cases
    if market_price <= 0:
        return 0.01
    if T <= 0:
        return 0.01
    if S <= 0 or K <= 0:
        return 0.01

    # Intrinsic value check (UNDISCOUNTED)
    if ot == "CE":
        intrinsic = max(0.0, S - K)
    else:
        intrinsic = max(0.0, K - S)

    # If market price is below intrinsic beyond tolerance, reject
    if market_price < intrinsic - tolerance:
        return 0.01

    # Starting guess: 20% volatility
    sigma = 0.20

    stable_count = 0
    for _ in range(max_iterations):
        prev_sigma = sigma

        bs_price = black_scholes_price(S, K, T, r, sigma, ot)
        diff = bs_price - market_price

        # Convergence in price space
        if abs(diff) < tolerance:
            return max(0.001, sigma)

        vega_raw = _raw_vega(S, K, T, r, sigma)

        # Divergence sentinel: don't hide failure behind clamping
        if abs(vega_raw) < 1e-10:
            if abs(diff) > tolerance:
                _emit_log(
                    "warning",
                    "IV Newton-Raphson aborted due to tiny vega with large price mismatch",
                    vega_raw=vega_raw,
                    diff=diff,
                    tolerance=tolerance,
                    S=S,
                    K=K,
                    T=T,
                    option_type=ot,
                    sigma=sigma,
                )
                return 0.0
            # if diff is already small-ish, accept current sigma
            return max(0.001, min(5.0, sigma))

        # Newton-Raphson update
        sigma = sigma - diff / vega_raw

        # Keep sigma in reasonable bounds (0.1% to 500%)
        sigma = max(0.001, min(5.0, sigma))

        # Stagnation break ONLY if also price-converged (defensive; should rarely trigger due to earlier return)
        if abs(diff) < tolerance and abs(sigma - prev_sigma) < 1e-8:
            stable_count += 1
            if stable_count >= 2:
                break
        else:
            stable_count = 0

    return max(0.001, sigma)


# ==============================================
# Greeks Computation
# ==============================================

def compute_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "CE",
) -> float:
    """
    Compute option Delta.
    """
    ot = _validate_option_type(option_type)

    if T <= 0 or sigma <= 0:
        if ot == "CE":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = _compute_d1(S, K, T, r, sigma)

    if ot == "CE":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


def compute_gamma(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Compute option Gamma.
    Same for both CE and PE. Always positive.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0

    d1 = _compute_d1(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def compute_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "CE",
) -> float:
    """
    Compute option Theta (per day).
    """
    ot = _validate_option_type(option_type)

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _compute_d1(S, K, T, r, sigma)
    d2 = _compute_d2(d1, sigma, T)

    term1 = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))

    if ot == "CE":
        theta_annual = term1 - r * K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        theta_annual = term1 + r * K * math.exp(-r * T) * _norm_cdf(-d2)

    return theta_annual / 365.0


def compute_vega(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Compute option Vega (per 1% IV change).
    """
    return _raw_vega(S, K, T, r, sigma) / 100.0


def compute_all_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "CE",
) -> Dict[str, float]:
    """
    Compute all Greeks at once.
    """
    ot = _validate_option_type(option_type)

    if T <= 0 or sigma <= 0:
        return {
            "delta": compute_delta(S, K, T, r, sigma, ot),
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

    d1 = _compute_d1(S, K, T, r, sigma)
    d2 = _compute_d2(d1, sigma, T)
    sqrt_t = math.sqrt(T)
    pdf_d1 = _norm_pdf(d1)
    exp_rt = math.exp(-r * T)

    if ot == "CE":
        delta = _norm_cdf(d1)
    else:
        delta = _norm_cdf(d1) - 1.0

    gamma = pdf_d1 / (S * sigma * sqrt_t) if S > 0 else 0.0

    term1 = -(S * pdf_d1 * sigma) / (2.0 * sqrt_t)
    if ot == "CE":
        theta_annual = term1 - r * K * exp_rt * _norm_cdf(d2)
    else:
        theta_annual = term1 + r * K * exp_rt * _norm_cdf(-d2)
    theta = theta_annual / 365.0

    vega = S * pdf_d1 * sqrt_t / 100.0

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
    }


# ==============================================
# Module Self-Test
# ==============================================

if __name__ == "__main__":
    print("=" * 60)
    print("  JUNIOR ALADDIN — Black-Scholes Module Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # Standard test parameters
    S = 24500.0
    K = 24500.0
    T = 3.0 / 365.0
    r = 0.065
    sigma = 0.15

    print(f"  Test Parameters:")
    print(f"    Spot (S)      = {S}")
    print(f"    Strike (K)    = {K}")
    print(f"    Time (T)      = {T:.6f} years ({T*365:.1f} days)")
    print(f"    Rate (r)      = {r}")
    print(f"    Volatility    = {sigma*100:.1f}%")
    print()

    # ── Test 1: BS Price - CE ──
    print("  [Test 1] Black-Scholes Price (CE)...")
    ce_price = black_scholes_price(S, K, T, r, sigma, "CE")
    print(f"    CE Price = {ce_price:.2f}")
    if 30 < ce_price < 200:
        print(f"    ✅ Price in reasonable range (30-200)")
        passed += 1
    else:
        print(f"    ❌ Price out of range")
        failed += 1

    # ── Test 2: BS Price - PE ──
    print("\n  [Test 2] Black-Scholes Price (PE)...")
    pe_price = black_scholes_price(S, K, T, r, sigma, "PE")
    print(f"    PE Price = {pe_price:.2f}")
    if 30 < pe_price < 200:
        print(f"    ✅ Price in reasonable range (30-200)")
        passed += 1
    else:
        print(f"    ❌ Price out of range")
        failed += 1

    # ── Test 3: Put-Call Parity ──
    print("\n  [Test 3] Put-Call Parity Check...")
    parity_left = ce_price - pe_price
    parity_right = S - K * math.exp(-r * T)
    parity_diff = abs(parity_left - parity_right)
    print(f"    C - P       = {parity_left:.4f}")
    print(f"    S - Ke^-rT  = {parity_right:.4f}")
    print(f"    Difference  = {parity_diff:.6f}")
    if parity_diff < 0.01:
        print(f"    ✅ Put-Call parity holds (diff < 0.01)")
        passed += 1
    else:
        print(f"    ❌ Parity violated")
        failed += 1

    # ── Test 4: Expired Option ──
    print("\n  [Test 4] Expired Option (T=0)...")
    exp_itm_ce = black_scholes_price(24600, 24500, 0, r, sigma, "CE")
    exp_otm_ce = black_scholes_price(24400, 24500, 0, r, sigma, "CE")
    print(f"    ITM CE (S=24600, K=24500, T=0) = {exp_itm_ce:.2f}")
    print(f"    OTM CE (S=24400, K=24500, T=0) = {exp_otm_ce:.2f}")
    if exp_itm_ce == 100.0 and exp_otm_ce == 0.0:
        print(f"    ✅ Correct intrinsic values at expiry")
        passed += 1
    else:
        print(f"    ❌ Expected ITM=100, OTM=0")
        failed += 1

    # ── Test 5: Implied Volatility ──
    print("\n  [Test 5] Implied Volatility...")
    market_ltp = 100.0
    iv = implied_volatility(market_ltp, S, K, T, r, "CE")
    print(f"    Market LTP = {market_ltp:.2f}")
    print(f"    Computed IV = {iv*100:.2f}%")
    verify_price = black_scholes_price(S, K, T, r, iv, "CE")
    print(f"    Verify Price = {verify_price:.2f} (should be close to {market_ltp})")
    if abs(verify_price - market_ltp) < 0.50:
        print(f"    ✅ IV correctly recovers market price (diff < 0.50)")
        passed += 1
    else:
        print(f"    ❌ IV recovery failed (diff = {abs(verify_price - market_ltp):.4f})")
        failed += 1

    # ── Test 6: IV with different prices ──
    print("\n  [Test 6] IV for different market prices...")
    test_prices = [50.0, 100.0, 150.0, 200.0]
    iv_ok = True
    for mp in test_prices:
        iv_test = implied_volatility(mp, S, K, T, r, "CE")
        vp = black_scholes_price(S, K, T, r, iv_test, "CE")
        diff = abs(vp - mp)
        status = "✅" if diff < 0.50 else "❌"
        print(f"    {status} LTP={mp:.0f} -> IV={iv_test*100:.1f}% -> BS={vp:.2f} (diff={diff:.3f})")
        if diff >= 0.50:
            iv_ok = False
    if iv_ok:
        print(f"    ✅ All IV computations accurate")
        passed += 1
    else:
        print(f"    ❌ Some IV computations inaccurate")
        failed += 1

    # ── Test 7: IV for PE ──
    print("\n  [Test 7] IV for Put Option...")
    pe_market = 95.0
    iv_pe = implied_volatility(pe_market, S, K, T, r, "PE")
    vp_pe = black_scholes_price(S, K, T, r, iv_pe, "PE")
    print(f"    PE Market = {pe_market:.2f}")
    print(f"    PE IV = {iv_pe*100:.2f}%")
    print(f"    PE Verify = {vp_pe:.2f}")
    if abs(vp_pe - pe_market) < 0.50:
        print(f"    ✅ PE IV computation accurate")
        passed += 1
    else:
        print(f"    ❌ PE IV inaccurate (diff = {abs(vp_pe - pe_market):.4f})")
        failed += 1

    # ── Test 8: Delta ──
    print("\n  [Test 8] Delta...")
    delta_ce = compute_delta(S, K, T, r, sigma, "CE")
    delta_pe = compute_delta(S, K, T, r, sigma, "PE")
    print(f"    ATM CE Delta = {delta_ce:.4f}")
    print(f"    ATM PE Delta = {delta_pe:.4f}")
    delta_sum = delta_ce + abs(delta_pe)
    if 0.45 < delta_ce < 0.55 and -0.55 < delta_pe < -0.45:
        print(f"    ✅ ATM deltas correct (CE~0.50, PE~-0.50)")
        passed += 1
    else:
        print(f"    ❌ Deltas out of range")
        failed += 1
    if abs(delta_sum - 1.0) < 0.02:
        print(f"    ✅ |CE Delta| + |PE Delta| = {delta_sum:.4f} (close to 1.0)")
        passed += 1
    else:
        print(f"    ❌ Delta sum should be close to 1.0 (got {delta_sum:.4f})")
        failed += 1

    # ── Test 9: ITM/OTM Delta ──
    print("\n  [Test 9] ITM/OTM Delta...")
    delta_itm = compute_delta(25000, 24500, T, r, sigma, "CE")
    delta_otm = compute_delta(24000, 24500, T, r, sigma, "CE")
    print(f"    Deep ITM CE Delta (S=25000) = {delta_itm:.4f}")
    print(f"    Deep OTM CE Delta (S=24000) = {delta_otm:.4f}")
    if delta_itm > 0.90 and delta_otm < 0.10:
        print(f"    ✅ ITM delta > 0.90, OTM delta < 0.10")
        passed += 1
    else:
        print(f"    ❌ ITM/OTM deltas incorrect")
        failed += 1

    # ── Test 10: Gamma ──
    print("\n  [Test 10] Gamma...")
    gamma = compute_gamma(S, K, T, r, sigma)
    print(f"    ATM Gamma = {gamma:.6f}")
    if gamma > 0:
        print(f"    ✅ Gamma is positive")
        passed += 1
    else:
        print(f"    ❌ Gamma should be positive")
        failed += 1

    # ── Test 11: Theta ──
    print("\n  [Test 11] Theta...")
    theta_ce = compute_theta(S, K, T, r, sigma, "CE")
    theta_pe = compute_theta(S, K, T, r, sigma, "PE")
    print(f"    ATM CE Theta = {theta_ce:.2f} per day")
    print(f"    ATM PE Theta = {theta_pe:.2f} per day")
    if theta_ce < 0:
        print(f"    ✅ CE Theta is negative (time decay)")
        passed += 1
    else:
        print(f"    ❌ CE Theta should be negative")
        failed += 1

    # ── Test 12: Vega ──
    print("\n  [Test 12] Vega...")
    vega = compute_vega(S, K, T, r, sigma)
    print(f"    ATM Vega = {vega:.2f} per 1% IV change")
    if vega > 0:
        print(f"    ✅ Vega is positive")
        passed += 1
    else:
        print(f"    ❌ Vega should be positive")
        failed += 1

    # ── Test 13: All Greeks at once ──
    print("\n  [Test 13] compute_all_greeks()...")
    greeks = compute_all_greeks(S, K, T, r, sigma, "CE")
    print(f"    Delta = {greeks['delta']:.4f}")
    print(f"    Gamma = {greeks['gamma']:.6f}")
    print(f"    Theta = {greeks['theta']:.2f}")
    print(f"    Vega  = {greeks['vega']:.2f}")
    if (abs(greeks["delta"] - delta_ce) < 0.0001 and
        abs(greeks["gamma"] - gamma) < 0.000001 and
        abs(greeks["theta"] - theta_ce) < 0.01 and
        abs(greeks["vega"] - vega) < 0.01):
        print(f"    ✅ All Greeks consistent with individual functions")
        passed += 1
    else:
        print(f"    ❌ Greeks inconsistent")
        failed += 1

    # ── Test 14: IV round-trip verification ──
    print("\n  [Test 14] IV Round-Trip (BS price -> IV -> BS price)...")
    known_sigma = 0.18
    known_price = black_scholes_price(S, K, T, r, known_sigma, "CE")
    recovered_iv = implied_volatility(known_price, S, K, T, r, "CE")
    recovered_price = black_scholes_price(S, K, T, r, recovered_iv, "CE")
    print(f"    Known sigma = {known_sigma*100:.1f}%")
    print(f"    BS price    = {known_price:.2f}")
    print(f"    Recovered IV = {recovered_iv*100:.2f}%")
    print(f"    Recovered price = {recovered_price:.2f}")
    if abs(recovered_iv - known_sigma) < 0.001:
        print(f"    ✅ IV recovery accurate (diff = {abs(recovered_iv - known_sigma)*100:.3f}%)")
        passed += 1
    else:
        print(f"    ❌ IV recovery inaccurate")
        failed += 1

    # ── Test 15: Edge cases ──
    print("\n  [Test 15] Edge Cases...")
    edge_ok = True

    iv_zero = implied_volatility(0, S, K, T, r, "CE")
    print(f"    IV for price=0: {iv_zero:.4f} {'✅' if iv_zero == 0.01 else '❌'}")
    if iv_zero != 0.01:
        edge_ok = False

    deep_itm = black_scholes_price(25000, 24000, T, r, 0.15, "CE")
    print(f"    Deep ITM CE (S=25000, K=24000): {deep_itm:.2f} {'✅' if deep_itm > 990 else '❌'}")
    if deep_itm <= 990:
        edge_ok = False

    deep_otm = black_scholes_price(24000, 25000, T, r, 0.15, "CE")
    print(f"    Deep OTM CE (S=24000, K=25000): {deep_otm:.2f} {'✅' if deep_otm < 1.0 else '❌'}")
    if deep_otm >= 1.0:
        edge_ok = False

    if edge_ok:
        print(f"    ✅ All edge cases passed")
        passed += 1
    else:
        print(f"    ❌ Some edge cases failed")
        failed += 1

    # ── Test 16: Tail CDF should not saturate to exactly 1.0/0.0 ──
    print("\n  [Test 16] Tail CDF Precision (no saturation)...")
    c_hi = _norm_cdf(10.0)
    c_lo = _norm_cdf(-10.0)
    print(f"    CDF(10.0)  = {c_hi:.18f}")
    print(f"    CDF(-10.0) = {c_lo:.18f}")
    if c_hi < 1.0 and c_lo > 0.0:
        print("    ✅ Tail CDF does not saturate to exact 1.0/0.0")
        passed += 1
    else:
        print("    ❌ Tail CDF saturation detected")
        failed += 1

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n  🎉 Black-Scholes Module working perfectly!")
        print("  ✅ Ready for next module.")
    else:
        print(f"\n  ⚠️ {failed} tests failed.")
    print("=" * 60)