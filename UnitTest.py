import pytest
import numpy as np
from numpy import log, sqrt, exp
from scipy.stats import norm

# Import BlackScholes class
from black_scholes import BlackScholes  # Assuming the class is in black_scholes.py

class TestBlackScholes:
    @pytest.fixture
    def default_bs(self):
        """Fixture providing a default BlackScholes instance with typical values"""
        return BlackScholes(
            time_to_maturity=1.0,
            strike=100.0,
            current_price=100.0,
            volatility=0.2,
            interest_rate=0.05
        )

    def test_initialization(self, default_bs):
        """Test proper initialization of BlackScholes instance"""
        assert default_bs.time_to_maturity == 1.0
        assert default_bs.strike == 100.0
        assert default_bs.current_price == 100.0
        assert default_bs.volatility == 0.2
        assert default_bs.interest_rate == 0.05
        assert default_bs.call_price is None
        assert default_bs.put_price is None

    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate exceptions"""
        with pytest.raises(ValueError):
            BlackScholes(time_to_maturity=-1, strike=100, current_price=100, 
                        volatility=0.2, interest_rate=0.05)
        
        with pytest.raises(ValueError):
            BlackScholes(time_to_maturity=1, strike=-100, current_price=100, 
                        volatility=0.2, interest_rate=0.05)
        
        with pytest.raises(ValueError):
            BlackScholes(time_to_maturity=1, strike=100, current_price=-100, 
                        volatility=0.2, interest_rate=0.05)
        
        with pytest.raises(ValueError):
            BlackScholes(time_to_maturity=1, strike=100, current_price=100, 
                        volatility=-0.2, interest_rate=0.05)

    def test_at_the_money_call(self, default_bs):
        """Test at-the-money call option pricing"""
        call_price, _ = default_bs.calculate_prices()
        # For ATM options with 1 year to expiry, call should be around 10% of spot
        # with typical volatility and interest rate values
        expected_approx = default_bs.current_price * 0.1
        assert abs(call_price - expected_approx) < expected_approx * 0.3

    def test_at_the_money_put(self, default_bs):
        """Test at-the-money put option pricing"""
        _, put_price = default_bs.calculate_prices()
        # Put-call parity check: C - P = S - K*exp(-rT)
        call_price, _ = default_bs.calculate_prices()
        parity_value = (default_bs.current_price - 
                       default_bs.strike * exp(-default_bs.interest_rate * default_bs.time_to_maturity))
        assert abs((call_price - put_price) - parity_value) < 1e-10

    def test_deep_in_the_money_call(self):
        """Test deep in-the-money call option pricing"""
        bs = BlackScholes(time_to_maturity=1.0, strike=50.0, current_price=100.0,
                         volatility=0.2, interest_rate=0.05)
        call_price, _ = bs.calculate_prices()
        # Deep ITM call should be worth approximately S - K*exp(-rT)
        expected_min = bs.current_price - bs.strike * exp(-bs.interest_rate * bs.time_to_maturity)
        assert call_price > expected_min

    def test_deep_out_of_the_money_call(self):
        """Test deep out-of-the-money call option pricing"""
        bs = BlackScholes(time_to_maturity=1.0, strike=200.0, current_price=100.0,
                         volatility=0.2, interest_rate=0.05)
        call_price, _ = bs.calculate_prices()
        # Deep OTM call should be worth very little
        assert call_price < bs.current_price * 0.01

    def test_zero_volatility(self):
        """Test option pricing with zero volatility"""
        bs = BlackScholes(time_to_maturity=1.0, strike=100.0, current_price=100.0,
                         volatility=1e-10, interest_rate=0.05)
        with pytest.raises(ValueError):
            bs.calculate_prices()

    def test_call_delta(self):
        """Test call option delta calculation"""
        bs = BlackScholes(time_to_maturity=1.0, strike=100.0, current_price=100.0,
                         volatility=0.2, interest_rate=0.05)
        # Calculate delta numerically
        epsilon = 0.01
        bs_up = BlackScholes(time_to_maturity=1.0, strike=100.0, 
                           current_price=100.0 + epsilon,
                           volatility=0.2, interest_rate=0.05)
        bs_down = BlackScholes(time_to_maturity=1.0, strike=100.0, 
                             current_price=100.0 - epsilon,
                             volatility=0.2, interest_rate=0.05)
        
        call_up, _ = bs_up.calculate_prices()
        call_down, _ = bs_down.calculate_prices()
        numerical_delta = (call_up - call_down) / (2 * epsilon)
        
        # ATM call delta should be close to 0.5
        assert abs(numerical_delta - 0.5) < 0.1

    def test_put_call_parity(self):
        """Test put-call parity relationship"""
        test_cases = [
            {'S': 100, 'K': 100, 'T': 1, 'σ': 0.2, 'r': 0.05},
            {'S': 100, 'K': 110, 'T': 0.5, 'σ': 0.3, 'r': 0.02},
            {'S': 100, 'K': 90, 'T': 2, 'σ': 0.15, 'r': 0.03}
        ]
        
        for case in test_cases:
            bs = BlackScholes(
                time_to_maturity=case['T'],
                strike=case['K'],
                current_price=case['S'],
                volatility=case['σ'],
                interest_rate=case['r']
            )
            call_price, put_price = bs.calculate_prices()
            # C - P = S - K*exp(-rT)
            left_side = call_price - put_price
            right_side = case['S'] - case['K'] * exp(-case['r'] * case['T'])
            assert abs(left_side - right_side) < 1e-10

    def test_implied_volatility(self):
        """Test implied volatility calculation"""
        bs = BlackScholes(time_to_maturity=1.0, strike=100.0, current_price=100.0,
                         volatility=0.2, interest_rate=0.05)
        call_price, _ = bs.calculate_prices()
        
        # Calculate implied volatility using the known call price
        implied_vol = bs.implied_volatility(call_price, 'call')
        assert abs(implied_vol - 0.2) < 1e-4

    def test_profit_loss_calculation(self):
        """Test profit/loss calculation for options"""
        bs = BlackScholes(time_to_maturity=1.0, strike=100.0, current_price=100.0,
                         volatility=0.2, interest_rate=0.05)
        call_price, put_price = bs.calculate_prices()
        
        # Test call option P&L
        purchase_price_call = 10.0
        pl_call = bs.calculate_pl(purchase_price_call, 'call')
        assert pl_call == call_price - purchase_price_call
        
        # Test put option P&L
        purchase_price_put = 10.0
        pl_put = bs.calculate_pl(purchase_price_put, 'put')
        assert pl_put == put_price - purchase_price_put
        
        # Test invalid option type
        with pytest.raises(ValueError):
            bs.calculate_pl(10.0, 'invalid')

    def test_edge_cases(self):
        """Test edge cases for option pricing"""
        # Very short time to maturity
        bs = BlackScholes(time_to_maturity=1e-6, strike=100.0, current_price=100.0,
                         volatility=0.2, interest_rate=0.05)
        call_price, put_price = bs.calculate_prices()
        assert call_price >= 0
        assert put_price >= 0
        
        # Very high volatility
        bs = BlackScholes(time_to_maturity=1.0, strike=100.0, current_price=100.0,
                         volatility=1.0, interest_rate=0.05)
        call_price, put_price = bs.calculate_prices()
        assert call_price >= 0
        assert put_price >= 0
        
        # Very high interest rate
        bs = BlackScholes(time_to_maturity=1.0, strike=100.0, current_price=100.0,
                         volatility=0.2, interest_rate=0.5)
        call_price, put_price = bs.calculate_prices()
        assert call_price >= 0
        assert put_price >= 0

if __name__ == '__main__':
    pytest.main([__file__])