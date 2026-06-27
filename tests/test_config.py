from src import config


def test_weights_present_and_sane():
    assert 0 < config.W_SEMANTIC <= 1
    assert 0 < config.W_FEATURES <= 1
    assert config.REFERENCE_DATE.year == 2026


def test_keyword_lists_nonempty():
    assert "infosys" in config.SERVICES_FIRMS
    assert any("machine learning" in t for t in config.RELEVANT_TITLE_KEYWORDS)
    assert "pune" in config.TARGET_CITIES
    assert "marketing" in config.NEGATIVE_TITLE_KEYWORDS
