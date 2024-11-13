import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import log, sqrt, exp
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(
    page_title="Advanced Options Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BlackScholes:
    def __init__(self, time_to_maturity: float, strike: float, current_price: float, 
                 volatility: float, interest_rate: float):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.call_price = None
        self.put_price = None
        self._d1 = None
        self._d2 = None

    def _calculate_d1_d2(self):
        """Calculate d1 and d2 parameters for Black-Scholes formula"""
        if self.time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.strike <= 0 or self.current_price <= 0:
            raise ValueError("Strike price and current price must be positive")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")

        self._d1 = (
            log(self.current_price / self.strike) + 
            (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity
        ) / (self.volatility * sqrt(self.time_to_maturity))

        self._d2 = self._d1 - self.volatility * sqrt(self.time_to_maturity)

    def calculate_prices(self):
        """Calculate call and put option prices"""
        self._calculate_d1_d2()

        self.call_price = (
            self.current_price * norm.cdf(self._d1) - 
            self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(self._d2)
        )

        self.put_price = (
            self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-self._d2) - 
            self.current_price * norm.cdf(-self._d1)
        )

        return self.call_price, self.put_price

    def calculate_greeks(self):
        """Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)"""
        self._calculate_d1_d2()
        
        # Common terms
        sqrt_t = sqrt(self.time_to_maturity)
        exp_rt = exp(-self.interest_rate * self.time_to_maturity)
        n_d1 = norm.pdf(self._d1)
        n_d2 = norm.pdf(self._d2)
        
        # Delta
        call_delta = norm.cdf(self._d1)
        put_delta = call_delta - 1

        # Gamma (same for call and put)
        gamma = n_d1 / (self.current_price * self.volatility * sqrt_t)

        # Theta
        call_theta = (
            -self.current_price * n_d1 * self.volatility / (2 * sqrt_t) -
            self.interest_rate * self.strike * exp_rt * norm.cdf(self._d2)
        )
        put_theta = (
            -self.current_price * n_d1 * self.volatility / (2 * sqrt_t) +
            self.interest_rate * self.strike * exp_rt * norm.cdf(-self._d2)
        )

        # Vega (same for call and put)
        vega = self.current_price * sqrt_t * n_d1

        # Rho
        call_rho = self.strike * self.time_to_maturity * exp_rt * norm.cdf(self._d2)
        put_rho = -self.strike * self.time_to_maturity * exp_rt * norm.cdf(-self._d2)

        return {
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': gamma,
            'call_theta': call_theta,
            'put_theta': put_theta,
            'vega': vega,
            'call_rho': call_rho,
            'put_rho': put_rho
        }

    def calculate_pl(self, purchase_price, option_type):
        """Calculate profit/loss for options position"""
        if option_type == 'call':
            return self.call_price - purchase_price
        elif option_type == 'put':
            return self.put_price - purchase_price
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'")

    
def implied_volatility_newton(bs_model, market_price, option_type, max_iterations=100, precision=1e-5):
    """Calculate implied volatility using Newton-Raphson method with safeguards"""
    vol = 0.5  # Start with 50% volatility
    for i in range(max_iterations):
        bs_model.volatility = vol
        call_price, put_price = bs_model.calculate_prices()
        price = call_price if option_type == 'call' else put_price
        
        # Calculate derivative of price with respect to volatility (vega)
        diff = price - market_price
        if abs(diff) < precision:
            return vol
            
        vega = bs_model.calculate_greeks()['vega']
        if abs(vega) < 1e-10:  # Avoid division by zero
            return None
            
        new_vol = vol - diff / vega
        
        # Apply bounds and dampening
        new_vol = max(0.001, min(5.0, new_vol))
        if abs(new_vol - vol) < precision:
            return vol
            
        vol = new_vol
    
    return None

def plot_greeks_sensitivity_analysis(bs_model):
    """Create sensitivity plots for Greeks analysis"""
    spot_range = np.linspace(bs_model.current_price * 0.5, bs_model.current_price * 1.5, 50)
    vol_range = np.linspace(0.1, 0.8, 50)
    
    # Create empty figures for each Greek
    delta_fig = go.Figure()
    gamma_fig = go.Figure()
    theta_fig = go.Figure()
    vega_fig = go.Figure()
    rho_fig = go.Figure()
    
    # Calculate values for each spot price and volatility combination
    X, Y = np.meshgrid(spot_range, vol_range)
    Z_delta = np.zeros_like(X)
    Z_gamma = np.zeros_like(X)
    Z_theta = np.zeros_like(X)
    Z_vega = np.zeros_like(X)
    Z_rho = np.zeros_like(X)
    
    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=bs_model.strike,
                current_price=spot_range[j],
                volatility=vol_range[i],
                interest_rate=bs_model.interest_rate
            )
            greeks = bs_temp.calculate_greeks()
            Z_delta[i, j] = greeks['call_delta']
            Z_gamma[i, j] = greeks['gamma']
            Z_theta[i, j] = greeks['call_theta']
            Z_vega[i, j] = greeks['vega']
            Z_rho[i, j] = greeks['call_rho']
    
    # Create surface plots
    delta_fig.add_trace(go.Surface(x=X, y=Y, z=Z_delta))
    gamma_fig.add_trace(go.Surface(x=X, y=Y, z=Z_gamma))
    theta_fig.add_trace(go.Surface(x=X, y=Y, z=Z_theta))
    vega_fig.add_trace(go.Surface(x=X, y=Y, z=Z_vega))
    rho_fig.add_trace(go.Surface(x=X, y=Y, z=Z_rho))
    
    # Update layouts
    for fig, title in [(delta_fig, "Delta"), (gamma_fig, "Gamma"), 
                      (theta_fig, "Theta"), (vega_fig, "Vega"), 
                      (rho_fig, "Rho")]:
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Spot Price",
                yaxis_title="Volatility",
                zaxis_title=title
            ),
            height=500
        )
    
    return delta_fig, gamma_fig, theta_fig, vega_fig, rho_fig


def plot_iv_surface(symbol):
    """Create a 3D surface plot of implied volatilities across strikes and expiries"""
    try:
        # Get market data
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period="1d")["Close"].iloc[-1]
        risk_free_rate = 0.05
        
        # Get all options data
        iv_data = []
        expiry_dates = ticker.options[:100]  # Limit to first 6 expiration dates
        
        for expiry in expiry_dates:
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            days_to_expiry = (expiry_date - datetime.now()).days
            time_to_maturity = days_to_expiry / 365
            
            options = ticker.option_chain(expiry)
            
            for chain, option_type in [(options.calls, 'call'), (options.puts, 'put')]:
                for _, row in chain.iterrows():
                    if row['lastPrice'] > 0 and not pd.isna(row['lastPrice']):
                        try:
                            bs = BlackScholes(
                                time_to_maturity=max(time_to_maturity, 0.01),
                                strike=row['strike'],
                                current_price=current_price,
                                volatility=0.3,
                                interest_rate=risk_free_rate
                            )
                            
                            # Use implied_volatility_newton instead of implied_volatility
                            iv = implied_volatility_newton(bs, row['lastPrice'], option_type)
                            
                            if iv is not None and 0.0001 <= iv <= 2.0:
                                moneyness = row['strike'] / current_price
                                iv_data.append({
                                    'Time_To_Maturity': time_to_maturity,
                                    'Moneyness': moneyness,
                                    'IV': iv,
                                    'Option_Type': option_type
                                })
                        except Exception as e:
                            continue
        
        if not iv_data:
            return None
            
        iv_df = pd.DataFrame(iv_data)
        
        fig = go.Figure()
        
        for option_type in ['call', 'put']:
            option_data = iv_df[iv_df['Option_Type'] == option_type]
            
            if not option_data.empty:
                fig.add_trace(go.Mesh3d(
                    x=option_data['Time_To_Maturity'],
                    y=option_data['Moneyness'],
                    z=option_data['IV'],
                    name=f'{option_type.capitalize()} IV',
                    opacity=0.8,
                    colorscale='Viridis',
                    showscale=True
                ))
        
        fig.update_layout(
            title=f"Implied Volatility Surface - {symbol}",
            scene=dict(
                xaxis_title="Time to Maturity (Years)",
                yaxis_title="Moneyness (Strike/Spot)",
                zaxis_title="Implied Volatility",
            ),
            height=800,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating IV surface: {e}")
        return None

# 4. Handle potential NaN values in calculations
def calculate_iv(row, current_price, risk_free_rate, days_to_expiry):
    """Calculate implied volatility with proper error handling"""
    try:
        if row['lastPrice'] <= 0 or pd.isna(row['lastPrice']) or row['volume'] == 0:
            return None
            
        bs = BlackScholes(
            time_to_maturity=max(days_to_expiry/365, 0.01),
            strike=row['strike'],
            current_price=current_price,
            volatility=0.3,
            interest_rate=risk_free_rate
        )
        
        option_type = 'call' if row['Option Type'] == 'Call' else 'put'
        return implied_volatility_newton(bs, row['lastPrice'], option_type)
        
    except Exception as e:
        return None
    
def plot_option_payoff(spot_prices, strike, premium, option_type):
    """Create an interactive payoff diagram using plotly"""
    payoffs = []
    profits = []
    
    if option_type == 'call':
        payoffs = np.maximum(spot_prices - strike, 0)
    else:  # put
        payoffs = np.maximum(strike - spot_prices, 0)
    
    profits = payoffs - premium
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_prices, y=profits, name='Profit/Loss',
                            line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=spot_prices, y=payoffs, name='Payoff',
                            line=dict(color='green')))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=strike, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f"{option_type.capitalize()} Option Payoff Diagram",
        xaxis_title="Spot Price",
        yaxis_title="Profit/Loss",
        hovermode='x unified'
    )
    
    return fig

 

def get_market_data(symbol):
    """Fetch current market data and available options data"""
    try:
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period="1d")["Close"].iloc[-1]
        risk_free_rate = 0.05  # You might want to fetch this from a reliable source
        
        # Get available options dates
        options_dates = ticker.options
        
        return {
            'current_price': current_price,
            'risk_free_rate': risk_free_rate,
            'options_dates': options_dates
        }
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return None

def get_options_chain(symbol, expiry_date):
    """Fetch options chain data for a specific expiry date"""
    try:
        ticker = yf.Ticker(symbol)
        options = ticker.option_chain(expiry_date)
        
        calls = options.calls
        puts = options.puts
        
        # Clean and prepare the data
        calls['Option Type'] = 'Call'
        puts['Option Type'] = 'Put'
        
        # Combine calls and puts
        options_data = pd.concat([calls, puts])
        
        return options_data
    except Exception as e:
        st.error(f"Error fetching options chain: {e}")
        return None
def plot_heatmap(bs, spot_range, vol_range, strike, purchase_price_call, purchase_price_put):
    """
    Create P&L heatmaps for call and put options using user-defined ranges
    """
    # Create meshgrid for spot prices and volatilities
    spot_prices = np.linspace(spot_range[0], spot_range[1], 10)  # 10 points for better visualization
    volatilities = np.linspace(vol_range[0], vol_range[1], 10)  # 10 points for better visualization
    
    call_pl = np.zeros((len(volatilities), len(spot_prices)))
    put_pl = np.zeros((len(volatilities), len(spot_prices)))

    for i, vol in enumerate(volatilities):
        for j, spot in enumerate(spot_prices):
            bs_temp = BlackScholes(
                time_to_maturity=bs.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs.interest_rate
            )
            call_price, put_price = bs_temp.calculate_prices()
            call_pl[i, j] = bs_temp.calculate_pl(purchase_price_call, 'call')
            put_pl[i, j] = bs_temp.calculate_pl(purchase_price_put, 'put')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Create heatmaps with updated ranges
    sns.heatmap(call_pl, ax=ax1, cmap="RdYlGn", annot=True, fmt=".2f",
                xticklabels=[f"{x:.1f}" for x in spot_prices],
                yticklabels=[f"{x:.2f}" for x in volatilities])
    ax1.set_title("Call Option P&L Heatmap")
    ax1.set_xlabel("Spot Price")
    ax1.set_ylabel("Volatility")

    sns.heatmap(put_pl, ax=ax2, cmap="RdYlGn", annot=True, fmt=".2f",
                xticklabels=[f"{x:.1f}" for x in spot_prices],
                yticklabels=[f"{x:.2f}" for x in volatilities])
    ax2.set_title("Put Option P&L Heatmap")
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Volatility")

    plt.tight_layout()
    return fig

    sns.heatmap(call_pl, ax=ax1, cmap="RdYlGn", annot=True, fmt=".2f",
                xticklabels=[f"{x:.1f}" for x in spot_range],
                yticklabels=[f"{x:.2f}" for x in vol_range])
    ax1.set_title("Call Option P&L Heatmap")
    ax1.set_xlabel("Spot Price")
    ax1.set_ylabel("Volatility")

    sns.heatmap(put_pl, ax=ax2, cmap="RdYlGn", annot=True, fmt=".2f",
                xticklabels=[f"{x:.1f}" for x in spot_range],
                yticklabels=[f"{x:.2f}" for x in vol_range])
    ax2.set_title("Put Option P&L Heatmap")
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Volatility")

    plt.tight_layout()
    return fig   
# Helper functions for the tabs
def display_options_chain(options_chain):
    """Display the options chain data"""
    if options_chain is not None:
        st.dataframe(options_chain)

def display_volatility_smile(options_chain, market_data):
    """Display the volatility smile plot"""
    if options_chain is not None:
        fig = go.Figure()
        
        for opt_type in ['Call', 'Put']:
            data = options_chain[options_chain['Option Type'] == opt_type]
            if not data.empty and 'IV' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['strike'],
                    y=data['IV'],
                    name=f"{opt_type} IV",
                    mode='markers+lines'
                ))
        
        fig.update_layout(
            title="Implied Volatility Smile",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_volume_analysis(options_chain):
    """Display volume analysis plot"""
    if options_chain is not None:
        fig = go.Figure()
        
        for opt_type in ['Call', 'Put']:
            data = options_chain[options_chain['Option Type'] == opt_type]
            
            fig.add_trace(go.Bar(
                name=f"{opt_type} Volume",
                x=data['strike'],
                y=data['volume'],
                marker_color='blue' if opt_type == 'Call' else 'red',
                opacity=0.6
            ))
            
            fig.add_trace(go.Scatter(
                name=f"{opt_type} Open Interest",
                x=data['strike'],
                y=data['openInterest'],
                line=dict(color='green' if opt_type == 'Call' else 'orange')
            ))
        
        fig.update_layout(
            title="Volume and Open Interest Analysis",
            xaxis_title="Strike Price",
            yaxis_title="Contracts",
            barmode='group',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
# Main application
st.title("Advanced Options Calculator")

st.sidebar.title("Advanced Options Calculator")
    
# Sidebar navigation
page = st.sidebar.selectbox(
    "Select Tool",
    ["Options Calculator", "Implied Volatility", "Greeks Analysis"]
)
with st.sidebar:
        st.write("Created by:")
        linkedin_url = "https://www.linkedin.com/in/siddharth-kondubhatla-7603031ba/"
        st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">Siddharth Kondubhatla</a>', unsafe_allow_html=True)
if page == "Options Calculator":
    st.header("Black-Scholes Pricing Model")
    
    # Input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_price = st.number_input("Current Stock Price", value=100.0, min_value=0.01)
        strike = st.number_input("Strike Price", value=100.0, min_value=0.01)
        
    with col2:
        time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01)
        volatility = st.number_input("Volatility (%)", value=20.0, min_value=0.01) / 100
        
    with col3:
        interest_rate = st.number_input("Risk-free Rate (%)", value=5.0, min_value=0.01) / 100

    try:
        bs = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
        call_price, put_price = bs.calculate_prices()
        greeks = bs.calculate_greeks()
        
        # Display results
        st.subheader("Option Prices")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Call Option Price", f"${call_price:.2f}")
            call_purchase = st.number_input("Call Purchase Price", value=10.0, min_value=0.0, step=0.01)
        with col2:
            st.metric("Put Option Price", f"${put_price:.2f}")
            put_purchase = st.number_input("Put Purchase Price", value=10.0, min_value=0.0, step=0.01)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Call Option")
            st.write(f"Price: ${call_price:.2f}")
            pl_call = bs.calculate_pl(call_purchase, 'call')
            st.write(f"P&L: ${pl_call:.2f}")
            
        with col2:
            st.subheader("Put Option")
            st.write(f"Price: ${put_price:.2f}")
            pl_put = bs.calculate_pl(put_purchase, 'put')
            st.write(f"P&L: ${pl_put:.2f}")

        tab1, tab2 = st.tabs(["P&L Heatmaps", "Payoff Diagrams"])
    
        with tab1:
            st.subheader("P&L Heatmap Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Add sliders for volatility range
                vol_min = st.slider('Min Volatility for Heatmap (%)', 
                                  min_value=1, 
                                  max_value=100, 
                                  value=int(volatility*50),
                                  step=1) / 100.0
                
                vol_max = st.slider('Max Volatility for Heatmap (%)', 
                                  min_value=1, 
                                  max_value=100, 
                                  value=int(volatility*150),
                                  step=1) / 100.0

                # Add inputs for spot price range
                spot_min = st.number_input('Min Spot Price',
                                         min_value=0.01,
                                         max_value=current_price,
                                         value=current_price*0.8,
                                         step=0.01)
                
                spot_max = st.number_input('Max Spot Price',
                                         min_value=spot_min,
                                         value=current_price*1.2,
                                         step=0.01)
            
            with col2:
                # Add refresh button
                refresh_heatmap = st.button('Refresh Heatmap', key='refresh_heatmap')
            
            # Store the parameters in session state
            if 'heatmap_params' not in st.session_state:
                st.session_state.heatmap_params = {
                    'spot_min': spot_min,
                    'spot_max': spot_max,
                    'vol_min': vol_min,
                    'vol_max': vol_max
                }
            
            # Update heatmap only when refresh button is clicked
            if refresh_heatmap:
                st.session_state.heatmap_params = {
                    'spot_min': spot_min,
                    'spot_max': spot_max,
                    'vol_min': vol_min,
                    'vol_max': vol_max
                }
            
            # Display validation errors if any
            if spot_max <= spot_min:
                st.error("Maximum spot price must be greater than minimum spot price")
            elif vol_max <= vol_min:
                st.error("Maximum volatility must be greater than minimum volatility")
            else:
                try:
                    # Create heatmap with stored parameters
                    fig = plot_heatmap(bs, 
                                     spot_range=(st.session_state.heatmap_params['spot_min'],
                                               st.session_state.heatmap_params['spot_max']),
                                     vol_range=(st.session_state.heatmap_params['vol_min'],
                                              st.session_state.heatmap_params['vol_max']),
                                     strike=strike,
                                     purchase_price_call=call_purchase,
                                     purchase_price_put=put_purchase)
                    st.pyplot(fig)
                    
                    st.markdown("""
                    **Understanding the Heatmap:**
                    - Green indicates profitable positions
                    - Red indicates losing positions
                    - Values show the theoretical P&L for each combination of spot price and volatility
                    
                    """)
                    
                except Exception as e:
                    st.error(f"Error generating heatmap: {str(e)}")

        with tab2:
            # Plot payoff diagrams
            spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
        
            col1, col2 = st.columns(2)
            with col1:
                call_payoff = plot_option_payoff(spot_range, strike, call_price, 'call')
                st.plotly_chart(call_payoff, use_container_width=True)
        
            with col2:
                put_payoff = plot_option_payoff(spot_range, strike, put_price, 'put')
                st.plotly_chart(put_payoff, use_container_width=True)

        # Display Greeks
        st.subheader("Greeks")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Call Delta", f"{greeks['call_delta']:.4f}")
            st.metric("Put Delta", f"{greeks['put_delta']:.4f}")
        
        with col2:
            st.metric("Gamma", f"{greeks['gamma']:.4f}")
            st.metric("Vega", f"{greeks['vega']:.4f}")
        
        with col3:
            st.metric("Call Theta", f"{greeks['call_theta']:.4f}")
            st.metric("Put Theta", f"{greeks['put_theta']:.4f}")
        
        with col4:
            st.metric("Call Rho", f"{greeks['call_rho']:.4f}")
            st.metric("Put Rho", f"{greeks['put_rho']:.4f}")

    except Exception as e:
        st.error(f"Error in calculation: {e}")
        
        
elif page == "Implied Volatility":
    st.header("Market Data & Volatility Analysis")
    
    # Create two columns for input
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
    
    with input_col2:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)", 
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Enter the risk-free rate as a percentage (e.g., 5.0 for 5%)"
        ) / 100
    
    if symbol:
        try:
            market_data = get_market_data(symbol)
            
            if market_data:
                # Market Overview Section
                st.subheader("Market Overview")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Stock Price", f"${market_data['current_price']:.2f}")
                
                
                # Display options chain
                maturity_date = st.selectbox("Select Maturity Date", market_data['options_dates'])
                
                if maturity_date:
                    options_chain = get_options_chain(symbol, maturity_date)
                    
                    if options_chain is not None:
                        # Calculate days until expiration
                        expiry = datetime.strptime(maturity_date, '%Y-%m-%d')
                        days_to_expiry = (expiry - datetime.now()).days
                        
                        # Add implied volatility calculation
                        current_price = market_data['current_price']
                        
                        # Calculate IV with progress bar
                        
                        progress_bar = st.progress(0)
                        
                        total_rows = len(options_chain)
                        options_chain['IV'] = None
                        
                        for idx, row in options_chain.iterrows():
                            try:
                                if row['lastPrice'] <= 0 or pd.isna(row['lastPrice']) or row['volume'] == 0:
                                    options_chain.at[idx, 'IV'] = None
                                else:
                                    iv = calculate_iv(
                                        row=row,
                                        current_price=current_price,
                                        risk_free_rate=risk_free_rate,
                                        days_to_expiry=days_to_expiry
                                    )
                                    options_chain.at[idx, 'IV'] = iv
                            except Exception as calc_error:
                                options_chain.at[idx, 'IV'] = None
                            finally:
                                progress_bar.progress((idx + 1) / total_rows)
                        
                        progress_bar.empty()
                        
                        
                        # Create tabs for different analyses
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Options Chain",
                            "Volatility Surface",
                            "Volatility Smile",
                            "Volume Analysis"
                        ])
                        
                        with tab1:
                            st.subheader("Options Chain")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                min_strike = float(options_chain['strike'].min())
                                max_strike = float(options_chain['strike'].max())
                                strike_range = st.slider(
                                    "Strike Price Range",
                                    min_value=min_strike,
                                    max_value=max_strike,
                                    value=(min_strike, max_strike)
                                )
                            
                            with col2:
                                option_type = st.multiselect(
                                    "Option Type",
                                    ['Call', 'Put'],
                                    default=['Call', 'Put']
                                )
                            
                            filtered_chain = options_chain[
                                (options_chain['strike'].between(strike_range[0], strike_range[1])) &
                                (options_chain['Option Type'].isin(option_type))
                            ]
                            
                            # Display formatted DataFrame
                            display_df = filtered_chain[[
                                'Option Type', 'strike', 'lastPrice', 'bid', 'ask',
                                'volume', 'openInterest', 'IV'
                            ]].copy()
                            
                            # Format the display DataFrame
                            def format_price(x):
                                return f"${x:.2f}" if pd.notnull(x) and x != 0 else "-"
                                
                            def format_iv(x):
                                return f"{x:.2%}" if pd.notnull(x) else "-"
                                
                            def format_number(x):
                                return f"{int(x):,}" if pd.notnull(x) and x != 0 else "-"
                            
                            display_df['strike'] = display_df['strike'].apply(format_price)
                            display_df['lastPrice'] = display_df['lastPrice'].apply(format_price)
                            display_df['bid'] = display_df['bid'].apply(format_price)
                            display_df['ask'] = display_df['ask'].apply(format_price)
                            display_df['volume'] = display_df['volume'].apply(format_number)
                            display_df['openInterest'] = display_df['openInterest'].apply(format_number)
                            display_df['IV'] = display_df['IV'].apply(format_iv)
                            
                            st.dataframe(display_df)
                        
                        with tab2:
                            st.subheader("Implied Volatility Surface")
                            with st.spinner(""):
                                iv_surface = plot_iv_surface(symbol)
                                if iv_surface is not None:
                                    st.plotly_chart(iv_surface, use_container_width=True)
                                    
                                    st.markdown("""
                                    **Understanding the IV Surface:**
                                    - X-axis: Time to Maturity shows how IV varies across different expiration dates
                                    - Y-axis: Moneyness (Strike/Spot) indicates if options are ITM, ATM, or OTM
                                    - Z-axis: Implied Volatility level
                                    - Color intensity represents the IV level
                                    """)
                        
                        with tab3:
                            st.subheader("Volatility Smile")
                            fig = go.Figure()
                            
                            for opt_type in ['Call', 'Put']:
                                data = filtered_chain[
                                    (filtered_chain['Option Type'] == opt_type) &
                                    (filtered_chain['IV'].notna())
                                ]
                                fig.add_trace(go.Scatter(
                                    x=data['strike'],
                                    y=data['IV'],
                                    name=f"{opt_type} IV",
                                    mode='markers+lines'
                                ))
                            
                            fig.update_layout(
                                title="Implied Volatility Smile",
                                xaxis_title="Strike Price",
                                yaxis_title="Implied Volatility",
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **Understanding the Volatility Smile:**
                            - Shows how implied volatility varies across different strike prices
                            - Typically forms a "smile" or "smirk" pattern
                            - ATM options usually have lower IV than OTM options
                            """)
                        
                        with tab4:
                            st.subheader("Volume Analysis")
                            fig = go.Figure()
                            
                            for opt_type in ['Call', 'Put']:
                                data = filtered_chain[filtered_chain['Option Type'] == opt_type]
                                
                                fig.add_trace(go.Bar(
                                    name=f"{opt_type} Volume",
                                    x=data['strike'],
                                    y=data['volume'],
                                    marker_color='blue' if opt_type == 'Call' else 'red',
                                    opacity=0.6
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    name=f"{opt_type} Open Interest",
                                    x=data['strike'],
                                    y=data['openInterest'],
                                    line=dict(color='green' if opt_type == 'Call' else 'orange')
                                ))
                            
                            fig.update_layout(
                                title="Volume and Open Interest Analysis",
                                xaxis_title="Strike Price",
                                yaxis_title="Contracts",
                                barmode='group',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **Understanding Volume Analysis:**
                            - Bars show trading volume for each strike price
                            - Lines show open interest (outstanding contracts)
                            - Higher values indicate more liquid options
                            """)
                        
        except Exception as error:
            st.error(f"Error in calculation: {str(error)}")

elif page == "Greeks Analysis":
    st.header("Greeks Analysis")
    
    try:
        # Input parameters for Greeks analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_price = st.number_input("Current Stock Price (Greeks)", value=100.0, min_value=0.01)
            strike = st.number_input("Strike Price (Greeks)", value=100.0, min_value=0.01)
        
        with col2:
            time_to_maturity = st.number_input("Time to Maturity (Years) (Greeks)", value=1.0, min_value=0.01)
            volatility = st.number_input("Volatility (%) (Greeks)", value=20.0, min_value=0.01) / 100
        
        with col3:
            interest_rate = st.number_input("Risk-free Rate (%) (Greeks)", value=5.0) / 100
            analysis_type = st.selectbox("Analysis Type", ["Surface Plots", "Sensitivity Analysis", "Greeks Evolution"])
        
        bs = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
        greeks = bs.calculate_greeks()
        
        if analysis_type == "Surface Plots":
            st.subheader("3D Surface Analysis of Greeks")
            
            delta_fig, gamma_fig, theta_fig, vega_fig, rho_fig = plot_greeks_sensitivity_analysis(bs)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Delta", "Gamma", "Theta", "Vega", "Rho"])
            
            with tab1:
                st.plotly_chart(delta_fig, use_container_width=True)
                st.write("Delta measures the rate of change of the option price with respect to the underlying asset price.")
            
            with tab2:
                st.plotly_chart(gamma_fig, use_container_width=True)
                st.write("Gamma measures the rate of change of Delta with respect to the underlying asset price.")
            
            with tab3:
                st.plotly_chart(theta_fig, use_container_width=True)
                st.write("Theta measures the rate of change of the option price with respect to time.")
            
            with tab4:
                st.plotly_chart(vega_fig, use_container_width=True)
                st.write("Vega measures the rate of change of the option price with respect to volatility.")
            
            with tab5:
                st.plotly_chart(rho_fig, use_container_width=True)
                st.write("Rho measures the rate of change of the option price with respect to the risk-free interest rate.")
        
        elif analysis_type == "Sensitivity Analysis":
            st.subheader("Greeks Sensitivity Analysis")
            
            sensitivity_param = st.selectbox("Vary Parameter", ["Spot Price", "Volatility", "Time to Maturity"])
            
            if sensitivity_param == "Spot Price":
                # Create spot price range for analysis
                spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
                greek_values = {
                    'Delta': [],
                    'Gamma': [],
                    'Theta': [],
                    'Vega': [],
                    'Rho': []
                }
                
                for spot in spot_range:
                    bs_temp = BlackScholes(time_to_maturity, strike, spot, volatility, interest_rate)
                    greeks = bs_temp.calculate_greeks()
                    greek_values['Delta'].append(greeks['call_delta'])
                    greek_values['Gamma'].append(greeks['gamma'])
                    greek_values['Theta'].append(greeks['call_theta'])
                    greek_values['Vega'].append(greeks['vega'])
                    greek_values['Rho'].append(greeks['call_rho'])
                
                fig = go.Figure()
                for greek, values in greek_values.items():
                    fig.add_trace(go.Scatter(x=spot_range, y=values, name=greek))
                
                fig.update_layout(
                    title="Greeks Sensitivity to Spot Price",
                    xaxis_title="Spot Price",
                    yaxis_title="Greek Value",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Greeks Evolution":
            st.subheader("Greeks Evolution Over Time")
            
            days_to_expiry = np.linspace(1, time_to_maturity * 365, 100)
            greek_values = {
                'Delta': [],
                'Gamma': [],
                'Theta': [],
                'Vega': [],
                'Rho': []
            }
            
            for days in days_to_expiry:
                bs_temp = BlackScholes(days/365, strike, current_price, volatility, interest_rate)
                greeks = bs_temp.calculate_greeks()
                greek_values['Delta'].append(greeks['call_delta'])
                greek_values['Gamma'].append(greeks['gamma'])
                greek_values['Theta'].append(greeks['call_theta'])
                greek_values['Vega'].append(greeks['vega'])
                greek_values['Rho'].append(greeks['call_rho'])
            
            fig = go.Figure()
            for greek, values in greek_values.items():
                fig.add_trace(go.Scatter(x=days_to_expiry, y=values, name=greek))
            
            fig.update_layout(
                title="Greeks Evolution Over Time",
                xaxis_title="Days to Expiry",
                yaxis_title="Greek Value",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in Greeks Analysis calculation: {str(e)}")
        st.error("Please check your input parameters and try again.")

