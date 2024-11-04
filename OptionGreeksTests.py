import pytest
import numpy as np
from numpy import log, sqrt, exp
from scipy.stats import norm

class TestBlackScholesGreeks:
    @pytest.fixture
    def default_bs(self):
        """Fixture providing a default BlackScholes instance"""
        return BlackScholes(
            time_to_maturity=1.0,
            strike=100.0,
            current_price=100.0,
            volatility=0.2,
            interest_rate=0.05
        )

    def calculate_numerical_derivative(self, bs, param, epsilon, greek_type='call'):
        """
        Calculate numerical derivative with respect to given parameter
        
        Parameters:
        - bs: BlackScholes instance
        - param: Parameter to differentiate with respect to ('S', 'K', 't', 'σ', 'r')
        - epsilon: Small change in parameter
        - greek_type: 'call' or 'put'
        """
        # Store original parameter value
        orig_value = getattr(bs, {
            'S': 'current_price',
            'K': 'strike',
            't': 'time_to_maturity',
            'σ': 'volatility',
            'r': 'interest_rate'
        }[param])

        # Calculate up price
        setattr(bs, {
            'S': 'current_price',
            'K': 'strike',
            't': 'time_to_maturity',
            'σ': 'volatility',
            'r': 'interest_rate'
        }[param], orig_value + epsilon)
        
        call_up, put_up = bs.calculate_prices()
        price_up = call_up if greek_type == 'call' else put_up

        # Calculate down price
        setattr(bs, {
            'S': 'current_price',
            'K': 'strike',
            't': 'time_to_maturity',
            'σ': 'volatility',
            'r': 'interest_rate'
        }[param], orig_value - epsilon)
        
        call_down, put_down = bs.calculate_prices()
        price_down = call_down if greek_type == 'call' else put_down

        # Reset parameter
        setattr(bs, {
            'S': 'current_price',
            'K': 'strike',
            't': 'time_to_maturity',
            'σ': 'volatility',
            'r': 'interest_rate'
        }[param], orig_value)

        return (price_up - price_down) / (2 * epsilon)

    def test_delta_call(self, default_bs):
        """Test call option Delta calculation"""
        # Calculate analytical Delta
        d1 = (log(default_bs.current_price / default_bs.strike) + 
              (default_bs.interest_rate + 0.5 * default_bs.volatility ** 2) * 
              default_bs.time_to_maturity) / (default_bs.volatility * 
              sqrt(default_bs.time_to_maturity))
        analytical_delta = norm.cdf(d1)
        
        # Calculate numerical Delta
        numerical_delta = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='call'
        )
        
        assert abs(analytical_delta - numerical_delta) < 1e-3
        
        # Test Delta bounds
        assert 0 <= analytical_delta <= 1
        
        # Test ATM Delta ≈ 0.5
        assert abs(analytical_delta - 0.5) < 0.1

    def test_delta_put(self, default_bs):
        """Test put option Delta calculation"""
        # Calculate analytical Delta
        d1 = (log(default_bs.current_price / default_bs.strike) + 
              (default_bs.interest_rate + 0.5 * default_bs.volatility ** 2) * 
              default_bs.time_to_maturity) / (default_bs.volatility * 
              sqrt(default_bs.time_to_maturity))
        analytical_delta = norm.cdf(d1) - 1
        
        # Calculate numerical Delta
        numerical_delta = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='put'
        )
        
        assert abs(analytical_delta - numerical_delta) < 1e-3
        
        # Test Delta bounds
        assert -1 <= analytical_delta <= 0
        
        # Test ATM Delta ≈ -0.5
        assert abs(analytical_delta + 0.5) < 0.1

    def test_gamma(self, default_bs):
        """Test Gamma calculation (same for calls and puts)"""
        # Calculate analytical Gamma
        d1 = (log(default_bs.current_price / default_bs.strike) + 
              (default_bs.interest_rate + 0.5 * default_bs.volatility ** 2) * 
              default_bs.time_to_maturity) / (default_bs.volatility * 
              sqrt(default_bs.time_to_maturity))
        
        analytical_gamma = norm.pdf(d1) / (default_bs.current_price * 
                                         default_bs.volatility * 
                                         sqrt(default_bs.time_to_maturity))
        
        # Calculate numerical Gamma using second derivative
        epsilon = 0.01
        delta_up = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon, greek_type='call'
        )
        
        default_bs.current_price -= 2 * epsilon
        delta_down = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon, greek_type='call'
        )
        
        numerical_gamma = (delta_up - delta_down) / (2 * epsilon)
        
        # Reset price
        default_bs.current_price += 2 * epsilon
        
        assert abs(analytical_gamma - numerical_gamma) < 1e-2
        
        # Test Gamma is positive
        assert analytical_gamma > 0

    def test_theta(self, default_bs):
        """Test Theta calculation"""
        # Calculate numerical Theta
        numerical_theta_call = -self.calculate_numerical_derivative(
            default_bs, 't', epsilon=1/365, greek_type='call'
        )
        
        numerical_theta_put = -self.calculate_numerical_derivative(
            default_bs, 't', epsilon=1/365, greek_type='put'
        )
        
        # Theta should be negative for calls and puts (usually)
        assert numerical_theta_call < 0
        assert numerical_theta_put < 0

    def test_vega(self, default_bs):
        """Test Vega calculation (same for calls and puts)"""
        # Calculate analytical Vega
        d1 = (log(default_bs.current_price / default_bs.strike) + 
              (default_bs.interest_rate + 0.5 * default_bs.volatility ** 2) * 
              default_bs.time_to_maturity) / (default_bs.volatility * 
              sqrt(default_bs.time_to_maturity))
        
        analytical_vega = default_bs.current_price * sqrt(default_bs.time_to_maturity) * \
                         norm.pdf(d1)
        
        # Calculate numerical Vega
        numerical_vega = self.calculate_numerical_derivative(
            default_bs, 'σ', epsilon=0.0001, greek_type='call'
        )
        
        assert abs(analytical_vega - numerical_vega) < 1e-2
        
        # Test Vega is positive
        assert analytical_vega > 0
        
        # Test Vega is the same for calls and puts
        numerical_vega_put = self.calculate_numerical_derivative(
            default_bs, 'σ', epsilon=0.0001, greek_type='put'
        )
        assert abs(numerical_vega - numerical_vega_put) < 1e-2

    def test_rho(self, default_bs):
        """Test Rho calculation"""
        # Calculate numerical Rho
        numerical_rho_call = self.calculate_numerical_derivative(
            default_bs, 'r', epsilon=0.0001, greek_type='call'
        )
        
        numerical_rho_put = self.calculate_numerical_derivative(
            default_bs, 'r', epsilon=0.0001, greek_type='put'
        )
        
        # Test Rho signs
        # Call Rho should be positive
        assert numerical_rho_call > 0
        # Put Rho should be negative
        assert numerical_rho_put < 0

    def test_greeks_special_cases(self):
        """Test option Greeks in special cases"""
        # Deep ITM call
        itm_bs = BlackScholes(
            time_to_maturity=1.0,
            strike=50.0,
            current_price=100.0,
            volatility=0.2,
            interest_rate=0.05
        )
        itm_delta = self.calculate_numerical_derivative(
            itm_bs, 'S', epsilon=0.01, greek_type='call'
        )
        assert abs(itm_delta - 1.0) < 0.1
        
        # Deep OTM call
        otm_bs = BlackScholes(
            time_to_maturity=1.0,
            strike=200.0,
            current_price=100.0,
            volatility=0.2,
            interest_rate=0.05
        )
        otm_delta = self.calculate_numerical_derivative(
            otm_bs, 'S', epsilon=0.01, greek_type='call'
        )
        assert abs(otm_delta) < 0.1
        
        # Near expiry option
        near_expiry_bs = BlackScholes(
            time_to_maturity=0.01,
            strike=100.0,
            current_price=100.0,
            volatility=0.2,
            interest_rate=0.05
        )
        near_expiry_gamma = self.calculate_numerical_derivative(
            near_expiry_bs, 'S', epsilon=0.01, greek_type='call'
        )
        assert near_expiry_gamma > 0
        
        # High volatility option
        high_vol_bs = BlackScholes(
            time_to_maturity=1.0,
            strike=100.0,
            current_price=100.0,
            volatility=0.5,
            interest_rate=0.05
        )
        high_vol_vega = self.calculate_numerical_derivative(
            high_vol_bs, 'σ', epsilon=0.0001, greek_type='call'
        )
        assert high_vol_vega > 0

    def test_delta_gamma_relationship(self, default_bs):
        """Test the relationship between Delta and Gamma"""
        # Gamma is the rate of change of Delta
        epsilon = 0.01
        spot = default_bs.current_price
        
        # Calculate Delta at three nearby spots
        default_bs.current_price = spot - epsilon
        delta1 = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='call'
        )
        
        default_bs.current_price = spot
        delta2 = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='call'
        )
        
        default_bs.current_price = spot + epsilon
        delta3 = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='call'
        )
        
        # Calculate Gamma numerically using these Deltas
        gamma_from_delta = (delta3 - delta1) / (2 * epsilon)
        
        # Calculate Gamma directly
        analytical_gamma = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='call'
        )
        
        assert abs(gamma_from_delta - analytical_gamma) < 1e-2

    def test_put_call_parity_greeks(self, default_bs):
        """Test put-call parity relationships for Greeks"""
        # Delta: Δcall - Δput = 1
        call_delta = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='call'
        )
        put_delta = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='put'
        )
        assert abs((call_delta - put_delta) - 1) < 1e-2
        
        # Gamma: γcall = γput
        call_gamma = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='call'
        )
        put_gamma = self.calculate_numerical_derivative(
            default_bs, 'S', epsilon=0.01, greek_type='put'
        )
        assert abs(call_gamma - put_gamma) < 1e-2
        
        # Vega: νcall = νput
        call_vega = self.calculate_numerical_derivative(
            default_bs, 'σ', epsilon=0.0001, greek_type='call'
        )
        put_vega = self.calculate_numerical_derivative(
            default_bs, 'σ', epsilon=0.0001, greek_type='put'
        )
        assert abs(call_vega - put_vega) < 1e-2

if __name__ == '__main__':
    pytest.main([__file__])