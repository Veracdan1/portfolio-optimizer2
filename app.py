import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Portfolio Optimizer",
    layout="wide"
)

st.title("Project 2 - Portfolio Optimizer")
st.write(
    "Enter 5-15 stock tickers, pull historical prices from Yahoo Finance, "
    "and compare optimized portfolios under different constraints and objectives."
)


# -----------------------------
# Helper functions
# -----------------------------
def parse_tickers(ticker_text: str):
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    # preserve order while removing duplicates
    seen = set()
    cleaned = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned


@st.cache_data(show_spinner=False)
def download_price_data(tickers, start_date, end_date):
    """
    Download adjusted historical prices from Yahoo Finance.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if data.empty:
        return pd.DataFrame()

    # yfinance can return different column structures depending on ticker count
    if len(tickers) == 1:
        # Single ticker case
        if "Close" in data.columns:
            prices = data[["Close"]].copy()
            prices.columns = tickers
        else:
            raise ValueError("Could not find Close prices in downloaded data.")
    else:
        if "Close" in data.columns:
            prices = data["Close"].copy()
        else:
            # fallback if structure differs
            possible_cols = [col for col in data.columns if isinstance(col, tuple) and col[0] == "Close"]
            if possible_cols:
                prices = data.loc[:, possible_cols].copy()
                prices.columns = [c[1] for c in possible_cols]
            else:
                raise ValueError("Could not find Close prices in downloaded data.")

    prices = prices.sort_index()
    prices = prices.ffill().dropna(axis=1, how="all").dropna()

    return prices


def annualized_stats(prices: pd.DataFrame):
    daily_returns = prices.pct_change().dropna()

    if daily_returns.empty:
        raise ValueError("Not enough data to calculate returns.")

    exp_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252

    return daily_returns, exp_returns, cov_matrix


def portfolio_return(weights, exp_returns):
    return float(np.dot(weights, exp_returns))


def portfolio_volatility(weights, cov_matrix):
    return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))


def portfolio_sharpe(weights, exp_returns, cov_matrix, risk_free_rate):
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return -np.inf
    ret = portfolio_return(weights, exp_returns)
    return (ret - risk_free_rate) / vol


def get_bounds(n_assets, max_weight, allow_short):
    if allow_short:
        # symmetric lower bound for a simple shorting option
        return tuple((-max_weight, max_weight) for _ in range(n_assets))
    return tuple((0, max_weight) for _ in range(n_assets))


def weight_sum_constraint():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1}


def target_return_constraint(exp_returns, target_return):
    return {"type": "eq", "fun": lambda w: portfolio_return(w, exp_returns) - target_return}


def optimize_max_sharpe(exp_returns, cov_matrix, risk_free_rate, max_weight, allow_short):
    n = len(exp_returns)
    x0 = np.array([1 / n] * n)
    bounds = get_bounds(n, max_weight, allow_short)
    constraints = [weight_sum_constraint()]

    def objective(w):
        return -portfolio_sharpe(w, exp_returns, cov_matrix, risk_free_rate)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result


def optimize_min_variance(exp_returns, cov_matrix, max_weight, allow_short):
    n = len(exp_returns)
    x0 = np.array([1 / n] * n)
    bounds = get_bounds(n, max_weight, allow_short)
    constraints = [weight_sum_constraint()]

    def objective(w):
        return portfolio_volatility(w, cov_matrix)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result


def optimize_target_return(exp_returns, cov_matrix, target_return, max_weight, allow_short):
    n = len(exp_returns)
    x0 = np.array([1 / n] * n)
    bounds = get_bounds(n, max_weight, allow_short)
    constraints = [
        weight_sum_constraint(),
        target_return_constraint(exp_returns, target_return)
    ]

    def objective(w):
        return portfolio_volatility(w, cov_matrix)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result


def build_efficient_frontier(exp_returns, cov_matrix, max_weight, allow_short, n_points=35):
    """
    Build efficient frontier by solving min-volatility portfolios
    across a range of target returns.
    """
    min_var_result = optimize_min_variance(exp_returns, cov_matrix, max_weight, allow_short)
    if not min_var_result.success:
        return pd.DataFrame()

    min_var_weights = min_var_result.x
    min_ret = portfolio_return(min_var_weights, exp_returns)

    # approximate upper return range from feasible single-asset and equal-weight ideas
    max_ret_estimate = float(exp_returns.max())
    if max_ret_estimate <= min_ret:
        max_ret_estimate = min_ret + 0.05

    target_returns = np.linspace(min_ret, max_ret_estimate, n_points)

    frontier = []
    for tr in target_returns:
        result = optimize_target_return(exp_returns, cov_matrix, tr, max_weight, allow_short)
        if result.success:
            w = result.x
            frontier.append({
                "Return": portfolio_return(w, exp_returns),
                "Volatility": portfolio_volatility(w, cov_matrix)
            })

    return pd.DataFrame(frontier)


def clean_weights(weights, threshold=1e-6):
    cleaned = np.where(np.abs(weights) < threshold, 0, weights)
    return cleaned


# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Inputs")

default_tickers = "AAPL, MSFT, NVDA, AMZN, GOOGL"
ticker_text = st.sidebar.text_input(
    "Enter 5-15 ticker symbols, separated by commas",
    value=default_tickers
)

col_dates_1, col_dates_2 = st.sidebar.columns(2)
start_date = col_dates_1.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = col_dates_2.date_input("End Date", value=pd.to_datetime("today"))

objective = st.sidebar.selectbox(
    "Optimization Objective",
    ["Max Sharpe", "Min Variance", "Target Return"]
)

risk_profile = st.sidebar.select_slider(
    "Risk Preference",
    options=["Conservative", "Balanced", "Aggressive"],
    value="Balanced"
)

# tie the risk profile to a default objective suggestion
if risk_profile == "Conservative":
    st.sidebar.caption("Conservative investors often prefer lower-volatility portfolios.")
elif risk_profile == "Balanced":
    st.sidebar.caption("Balanced investors often compare risk and return together.")
else:
    st.sidebar.caption("Aggressive investors may accept more volatility for higher expected return.")

max_weight = st.sidebar.slider(
    "Max Weight Per Stock",
    min_value=0.10,
    max_value=1.00,
    value=0.40,
    step=0.05
)

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate",
    min_value=0.00,
    max_value=0.10,
    value=0.04,
    step=0.005
)

allow_short = st.sidebar.checkbox("Allow Short Selling", value=False)

target_return = st.sidebar.slider(
    "Target Annual Return",
    min_value=0.00,
    max_value=0.40,
    value=0.12,
    step=0.01
)

run_button = st.sidebar.button("Optimize Portfolio")


# -----------------------------
# Main logic
# -----------------------------
if run_button:
    try:
        tickers = parse_tickers(ticker_text)

        if len(tickers) < 5 or len(tickers) > 15:
            st.error("Please enter between 5 and 15 tickers.")
            st.stop()

        # feasibility check for long-only case
        if not allow_short and max_weight * len(tickers) < 1:
            st.error(
                "The max weight is too low for the number of tickers. "
                "Increase max weight or add more tickers."
            )
            st.stop()

        with st.spinner("Downloading price data from Yahoo Finance..."):
            prices = download_price_data(tickers, start_date, end_date)

        if prices.empty:
            st.error("No price data was returned. Try different tickers or dates.")
            st.stop()

        valid_tickers = list(prices.columns)
        removed_tickers = [t for t in tickers if t not in valid_tickers]

        if len(valid_tickers) < 5:
            st.error(
                "After cleaning invalid or missing data, fewer than 5 usable tickers remained. "
                "Please choose a different set."
            )
            st.stop()

        if removed_tickers:
            st.warning(f"These tickers were removed due to missing/invalid data: {', '.join(removed_tickers)}")

        daily_returns, exp_returns, cov_matrix = annualized_stats(prices)

        st.subheader("1) Downloaded Price Data")
        st.write(f"Usable tickers: {', '.join(valid_tickers)}")
        st.dataframe(prices.tail())

        st.subheader("2) Estimated Inputs")
        summary_df = pd.DataFrame({
            "Expected Annual Return": exp_returns,
            "Annual Volatility": np.sqrt(np.diag(cov_matrix))
        })
        st.dataframe(summary_df.style.format("{:.2%}"))

        # choose optimization
        result = None
        if objective == "Max Sharpe":
            result = optimize_max_sharpe(
                exp_returns=exp_returns.values,
                cov_matrix=cov_matrix.values,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                allow_short=allow_short
            )
        elif objective == "Min Variance":
            result = optimize_min_variance(
                exp_returns=exp_returns.values,
                cov_matrix=cov_matrix.values,
                max_weight=max_weight,
                allow_short=allow_short
            )
        else:
            result = optimize_target_return(
                exp_returns=exp_returns.values,
                cov_matrix=cov_matrix.values,
                target_return=target_return,
                max_weight=max_weight,
                allow_short=allow_short
            )

        if result is None or not result.success:
            st.error(
                "Optimization failed. Try relaxing the max weight, changing the target return, "
                "or using a different ticker set."
            )
            st.stop()

        weights = clean_weights(result.x)

        portfolio_ret = portfolio_return(weights, exp_returns.values)
        portfolio_vol = portfolio_volatility(weights, cov_matrix.values)
        sharpe_ratio = (portfolio_ret - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else np.nan

        weights_df = pd.DataFrame({
            "Ticker": valid_tickers,
            "Optimal Weight": weights
        }).sort_values("Optimal Weight", ascending=False)

        st.subheader("3) Optimal Portfolio Output")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Expected Annual Return", f"{portfolio_ret:.2%}")
        metric_col2.metric("Expected Annual Risk (Volatility)", f"{portfolio_vol:.2%}")
        metric_col3.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")

        st.dataframe(weights_df.style.format({"Optimal Weight": "{:.2%}"}))

        st.subheader("4) Constraint / Risk Preference Summary")
        st.write(f"Objective selected: **{objective}**")
        st.write(f"Risk preference selected: **{risk_profile}**")
        st.write(f"Max weight per stock: **{max_weight:.0%}**")
        st.write(f"Short selling allowed: **{'Yes' if allow_short else 'No'}**")
        if objective == "Target Return":
            st.write(f"Target annual return: **{target_return:.2%}**")

        st.subheader("5) Efficient Frontier")

        frontier_df = build_efficient_frontier(
            exp_returns=exp_returns.values,
            cov_matrix=cov_matrix.values,
            max_weight=max_weight,
            allow_short=allow_short,
            n_points=40
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        if not frontier_df.empty:
            ax.plot(frontier_df["Volatility"], frontier_df["Return"], linewidth=2, label="Efficient Frontier")

        # plot current portfolio
        ax.scatter(
            portfolio_vol,
            portfolio_ret,
            s=120,
            marker="o",
            label=f"Selected Portfolio ({objective})"
        )

        # plot max sharpe and min variance reference portfolios
        ref_minvar = optimize_min_variance(exp_returns.values, cov_matrix.values, max_weight, allow_short)
        if ref_minvar.success:
            w_mv = ref_minvar.x
            ax.scatter(
                portfolio_volatility(w_mv, cov_matrix.values),
                portfolio_return(w_mv, exp_returns.values),
                s=100,
                marker="s",
                label="Min Variance"
            )

        ref_sharpe = optimize_max_sharpe(exp_returns.values, cov_matrix.values, risk_free_rate, max_weight, allow_short)
        if ref_sharpe.success:
            w_ms = ref_sharpe.x
            ax.scatter(
                portfolio_volatility(w_ms, cov_matrix.values),
                portfolio_return(w_ms, exp_returns.values),
                s=100,
                marker="^",
                label="Max Sharpe"
            )

        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Expected Return")
        ax.set_title("Efficient Frontier and Optimized Portfolio")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("6) Interpretation")
        st.write(
            "Use the controls in the sidebar to see how the portfolio changes when you adjust "
            "the objective, max-weight constraint, target return, or short-selling assumption."
        )
        st.write(
            "A lower max-weight cap usually forces more diversification, while a higher target return "
            "typically pushes the solution toward riskier allocations."
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Set your inputs in the sidebar and click **Optimize Portfolio**.")
