# distutils: language=c++

from libc.stdint cimport int64_t
from hummingbot.strategy.strategy_base cimport StrategyBase


cdef class SmartLiquidityStrategy(StrategyBase):
    cdef:
        object _market_info
        str _token
        object _order_amount
        bint _inventory_skew_enabled
        object _inventory_target_base_pct
        double _order_refresh_time
        object _order_refresh_tolerance_pct
        object _inventory_range_multiplier

        int _volatility_price_samples
        double _volatility_interval
        int _avg_volatility_samples
        object _volatility_to_spread_multiplier
        str _volatility_algorithm

        double _max_order_age
        object _max_spread
        double _status_report_interval
        bint _hb_app_notification

        int _buy_position
        int _sell_position
        object _buy_volume_in_front
        object _sell_volume_in_front
        int _calculated_buy_position
        int _calculated_sell_position
        object _ignore_over_spread
        double _filled_order_delay
        bint _should_wait_order_cancel_confirmation
        int _bits_behind

        double _last_timestamp

        double _cancel_timestamp
        double _create_timestamp

        int _filled_buy_orders_count
        int _filled_sell_orders_count
        bint _all_markets_ready
        int _refresh_time
        object _token_balances
        object _mid_price_buffer

        double _last_vol_reported
        object _sell_budget
        object _buy_budget
        object _ev_loop
        double _last_pnl_update
        object _last_vol
        int _start_time

        object _vol_indicator
        object _hb_app

        object _last_own_trade_price

        object _moving_price_band
        object _asset_price_delegate
        object _inventory_cost_price_delegate
        object _price_type

    cdef c_update_volatility(self)
    cdef object c_create_proposal_from_order_book_pos(self)
    cdef c_update_proposal_from_volatility(self, object proposal)
    cdef _c_vol_to_spread(self, object volatility)
    cdef tuple c_get_adjusted_available_balance(self, list orders)
    cdef object c_base_order_size(self, object price)
    cdef c_apply_budget_constraint(self, object proposal)
    cdef c_apply_inventory_skew(self, object proposal)
    cdef c_execute_proposal(self, object proposal)
    cdef c_did_complete_sell_order(self, object order_completed_event)
    cdef c_did_complete_buy_order(self, object order_completed_event)
    cdef c_cancel_active_orders_on_max_age_limit(self)
    cdef c_cancel_over_tolerance_orders(self, object proposal)
    cdef c_ignore_orders_below_min_amount(self, object proposal)
    cdef bint c_is_within_tolerance(self, list current_prices, list proposal_prices)
    cdef bint c_to_create_orders(self, object proposal)
    cdef set_timers(self)
    cdef c_apply_bits_behind(self, object proposal, int steps)
    cdef object c_get_mid_price(self)
    cdef c_apply_moving_price_band(self, object proposal)
