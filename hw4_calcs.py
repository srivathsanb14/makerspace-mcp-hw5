"""HW4 Q3 calculations, copied unchanged from HW4_Q3_HVAC_symbolic_discovery.ipynb (cell 3).
The MCP server in server.py is a thin typed wrapper over these."""
import math, json
import numpy as np

# ---------------------------------------------------------------- synthetic "smart thermostat" data
# Ground truth (hidden from the discovery tools): single-zone RC house model
TRUE = dict(UA_W_per_K=250.0, C_J_per_K=5.0e6, Qmax_W=5000.0, solar_m2=10.0,
            cop_eta=0.45)  # tau = C/UA = 20000 s = 5.56 h


def make_house_data(days=7, dt_min=5, seed=0, noise_K=0.05, T_mean=2.0):
    """One week of 5-minute thermostat logs: indoor T, outdoor T, heater duty u (0-1), solar W/m2."""
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / dt_min)
    t_h = np.arange(n) * dt_min / 60
    Tout = T_mean + 5 * np.sin(2 * np.pi * (t_h - 9) / 24) + 0.5 * np.cumsum(rng.normal(0, 0.05, n))
    solar = np.clip(600 * np.sin(2 * np.pi * (t_h - 6) / 24), 0, None) * rng.uniform(0.4, 1.0, n)
    a = TRUE['UA_W_per_K'] / TRUE['C_J_per_K'] * 3600       # 1/h
    b = TRUE['Qmax_W'] / TRUE['C_J_per_K'] * 3600           # K/h at full duty
    c = TRUE['solar_m2'] / TRUE['C_J_per_K'] * 3600         # K/h per W/m2
    T = np.empty(n); u = np.empty(n); T[0] = 19.0; integ = 0.0
    dt = dt_min / 60
    for k in range(n):
        hour = t_h[k] % 24
        Tset = 21.0 if 6 <= hour < 23 else 17.0
        err = Tset - T[k]; integ = np.clip(integ + err * dt, -5, 5)
        u[k] = np.clip(0.6 * err + 0.15 * integ + rng.normal(0, 0.08), 0, 1)
        if k < n - 1:
            dT = a * (Tout[k] - T[k]) + b * u[k] + c * solar[k]
            T[k + 1] = T[k] + dT * dt
    T_meas = T + rng.normal(0, noise_K, n)
    return dict(t_h=t_h, T_in=T_meas, T_out=Tout, u=u, solar=solar, dt_h=dt)


def make_cop_data(n=150, seed=1, noise=0.08):
    """Heat-pump test log: outdoor T, supply-water T, measured COP."""
    rng = np.random.default_rng(seed)
    Tout = rng.uniform(-15, 15, n)
    Tsup = rng.uniform(30, 55, n)
    Th, Tc = Tsup + 273.15, Tout + 273.15
    cop = TRUE['cop_eta'] * Th / (Th - Tc)
    return dict(T_out=Tout, T_supply=Tsup, COP=cop * (1 + rng.normal(0, noise / 3, n)))


HOUSE = make_house_data()
# A separate, colder week the discovery never sees: used only for validation.
HOLDOUT = make_house_data(seed=7, T_mean=-1.0)
COPDATA = make_cop_data()


# ---------------------------------------------------------------- tool 1: discover room thermal law
def calc_discover_thermal_model(threshold: float = 0.1, window_pts: int = 24):
    """Weak-form sparse symbolic regression (SINDy-style STLSQ) on the thermostat log.
    Picks which of 10 candidate terms drive dT_in/dt (K/h). Uses the integral form
    T(t+W) - T(t) = W * mean(terms over window) so sensor noise is averaged, not differentiated.
    Inputs: sparsity threshold in K/h per standard deviation of each term (0.01-1),
    window length in samples (6-72; 24 = 2 h at 5-min logging).
    Outputs: discovered equation, coefficients, R2, implied time constant tau (h),
    UA/C (1/h), heater gain (K/h at full duty), solar gain (K/h per W/m2)."""
    if not (0.01 <= threshold <= 1 and 6 <= window_pts <= 72):
        raise ValueError("threshold 0.01-1, window_pts 6-72")
    d = HOUSE
    Ti, To, u, s = d['T_in'], d['T_out'], d['u'], d['solar']
    lib = {'1': np.ones_like(Ti), 'T_in': Ti, 'T_out': To, 'u': u, 'solar': s,
           'T_in^2': Ti ** 2, 'T_out^2': To ** 2, 'T_in*T_out': Ti * To,
           'T_in*u': Ti * u, 'u^2': u ** 2}
    names = list(lib)
    F = np.column_stack([lib[n] for n in names])
    W = window_pts
    starts = np.arange(0, len(Ti) - W - 1, max(W // 4, 1))
    y = np.array([(Ti[a + W] - Ti[a]) / (W * d['dt_h']) for a in starts])
    X = np.array([F[a:a + W].mean(0) for a in starts])
    mu, sd = X.mean(0), X.std(0)
    mu[0], sd[0] = 0.0, 1.0
    Z = (X - mu) / sd
    w = np.linalg.lstsq(Z, y, rcond=None)[0]
    for _ in range(10):  # sequentially thresholded least squares
        small = np.abs(w) < threshold
        w[small] = 0
        big = ~small
        if big.any():
            w[big] = np.linalg.lstsq(Z[:, big], y, rcond=None)[0]
    coef = w / sd
    coef[0] = w[0] - (mu / sd)[1:] @ w[1:]
    pred = X @ coef
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    terms = {n: float(c) for n, c in zip(names, coef) if abs(c) > 1e-12}
    eq = "dT_in/dt = " + " + ".join(f"{c:.4g}*{n}" if n != '1' else f"{c:.4g}" for n, c in terms.items())
    out = {'equation_K_per_h': eq, 'coefficients': {n: round(c, 5) for n, c in terms.items()},
           'R2': round(float(r2), 4), 'n_terms': len(terms), 'n_windows': len(y), 'warnings': []}
    a_in, a_out = -terms.get('T_in', 0), terms.get('T_out', 0)
    out = {'equation_K_per_h': eq, 'coefficients': {n: round(c, 5) for n, c in terms.items()},
           'R2': round(float(r2), 4), 'n_terms': len(terms), 'warnings': []}
    if a_in > 0:
        out['UA_over_C_per_h'] = round(a_in, 4)
        out['time_constant_h'] = round(1 / a_in, 2)
        out['heater_gain_K_per_h'] = round(terms.get('u', 0), 3)
        out['solar_gain_K_per_h_per_Wm2'] = round(terms.get('solar', 0), 6)
        if a_out and abs(a_in - a_out) / a_in > 0.15:
            out['warnings'].append("T_in and T_out coefficients differ by >15%: not a clean (T_out - T_in) law.")
    extra = set(terms) - {'T_in', 'T_out', 'u', 'solar'}
    if extra:
        out['warnings'].append(f"Extra terms kept: {sorted(extra)}. Raise threshold if they are tiny.")
    return out


# ---------------------------------------------------------------- tool 2: discover heat-pump COP law
def _forms(To, Ts):
    Th, Tc, dT = Ts + 273.15, To + 273.15, Ts - To
    return {
        'COP = a': (np.ones_like(dT)[:, None], 1),
        'COP = a + b*dT': (np.column_stack([np.ones_like(dT), dT]), 2),
        'COP = a + b*dT + c*dT^2': (np.column_stack([np.ones_like(dT), dT, dT ** 2]), 3),
        'COP = a + b*T_out + c*T_supply': (np.column_stack([np.ones_like(dT), To, Ts]), 3),
        'COP = a/dT': ((1 / dT)[:, None], 1),
        'COP = a*T_hot/dT': ((Th / dT)[:, None], 1),
        'COP = a + b*T_hot/dT': (np.column_stack([np.ones_like(dT), Th / dT]), 2),
        'COP = a*exp(-b*dT)': (None, 2),
    }


def calc_discover_cop_law():
    """Symbolic search over 8 candidate COP formulas (constant, polynomial in lift dT,
    linear in both temps, 1/dT, Carnot fraction a*T_hot/dT, exponential), fitted by least squares
    on the heat-pump test log. Ranked by BIC so extra parameters must earn their place.
    Outputs: ranked table (formula, params, R2, RMSE, BIC), best formula, and if the best is a
    Carnot form the implied fraction of Carnot efficiency."""
    d = COPDATA
    To, Ts, y = d['T_out'], d['T_supply'], d['COP']
    n = len(y); rows = []
    for name, (X, k) in _forms(To, Ts).items():
        if X is None:  # exponential: fit log
            dT = Ts - To
            A = np.column_stack([np.ones_like(dT), -dT])
            p = np.linalg.lstsq(A, np.log(y), rcond=None)[0]
            pred = np.exp(p[0]) * np.exp(-p[1] * dT); params = [math.exp(p[0]), p[1]]
        else:
            params = np.linalg.lstsq(X, y, rcond=None)[0]; pred = X @ params
        rss = float(np.sum((y - pred) ** 2))
        rows.append({'formula': name, 'params': [round(float(v), 4) for v in params],
                     'R2': round(1 - rss / float(np.sum((y - y.mean()) ** 2)), 4),
                     'RMSE': round(math.sqrt(rss / n), 4),
                     'BIC': round(n * math.log(rss / n) + k * math.log(n), 1)})
    rows.sort(key=lambda r: r['BIC'])
    best = rows[0]
    out = {'ranked': rows, 'best_formula': best['formula'], 'best_params': best['params'],
           'n_points': n, 'T_out_range_C': [round(float(To.min()), 1), round(float(To.max()), 1)],
           'T_supply_range_C': [round(float(Ts.min()), 1), round(float(Ts.max()), 1)]}
    if best['formula'] == 'COP = a*T_hot/dT':
        out['fraction_of_carnot'] = best['params'][0]
    return out


# ---------------------------------------------------------------- tool 3: daily heating energy
def calc_daily_heating_energy(T_out_C: float, T_set_C: float = 21.0, T_supply_C: float = 40.0,
                         UA_W_per_K: float = None, fraction_of_carnot: float = None,
                         price_per_kWh: float = 0.17, solar_kWh_per_day: float = 0.0):
    """Steady-state daily heat demand and heat-pump electricity using the discovered laws.
    Q = UA*(T_set - T_out)*24 h - solar; electricity = Q / COP, COP = eta*T_hot/(T_hot - T_cold).
    Inputs: outdoor mean T (-30..20 C, below setpoint), setpoint (15-26 C), supply T (30-60 C),
    UA (50-2000 W/K; default from tool 1 x assumed C), Carnot fraction (0.2-0.7; default from tool 2),
    price ($/kWh), solar gains (kWh/day). Outputs: heat (kWh/day), COP, electricity (kWh/day), cost."""
    if UA_W_per_K is None:
        UA_W_per_K = calc_discover_thermal_model()['UA_over_C_per_h'] * TRUE['C_J_per_K'] / 3600
        src = "UA from discovered UA/C times assumed C = 5.0 MJ/K"
    else:
        src = "UA given"
    if fraction_of_carnot is None:
        fraction_of_carnot = calc_discover_cop_law().get('fraction_of_carnot', 0.45)
    if not (-30 <= T_out_C < T_set_C and 15 <= T_set_C <= 26 and 30 <= T_supply_C <= 60
            and 50 <= UA_W_per_K <= 2000 and 0.2 <= fraction_of_carnot <= 0.7):
        raise ValueError("Input outside valid range")
    Q = max(UA_W_per_K * (T_set_C - T_out_C) * 24 / 1000 - solar_kWh_per_day, 0)
    Th, Tc = T_supply_C + 273.15, T_out_C + 273.15
    cop = fraction_of_carnot * Th / (Th - Tc)
    E = Q / cop
    out = {'heat_demand_kWh_per_day': round(Q, 2), 'COP': round(cop, 2),
           'electricity_kWh_per_day': round(E, 2), 'cost_per_day': round(E * price_per_kWh, 2),
           'UA_used_W_per_K': round(UA_W_per_K, 1), 'carnot_fraction_used': round(fraction_of_carnot, 3),
           'note': src, 'warnings': []}
    rng = calc_discover_cop_law()
    if not rng['T_out_range_C'][0] <= T_out_C <= rng['T_out_range_C'][1]:
        out['warnings'].append("T_out outside the COP test range: extrapolating the discovered law.")
    return out


