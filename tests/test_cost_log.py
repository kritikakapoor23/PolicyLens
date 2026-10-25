from policylens.cost_log import estimate_cost_usd


def test_input_tokens_cost_matches_prices():
    model = "gemini-3.5-flash-lite"
    assert estimate_cost_usd(model, 1000000, 0) == 0


def test_zero_tokens_cost_nothing():
    assert estimate_cost_usd("gemini-3.5-flash-lite", 0, 0) == 0
