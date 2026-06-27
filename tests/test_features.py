# tests/test_features.py
from src import features as F

def test_experience_score_peaks_at_ideal():
    assert F.experience_score(7.0) == 1.0
    assert F.experience_score(7.0) > F.experience_score(3.0)
    assert F.experience_score(7.0) > F.experience_score(13.0)
    assert 0.0 <= F.experience_score(0.0) <= 1.0

def test_product_vs_services_share():
    services = [{"company": "Infosys", "duration_months": 24},
                {"company": "TCS", "duration_months": 24}]
    product = [{"company": "Flipkart", "duration_months": 24},
               {"company": "Swiggy", "duration_months": 24}]
    assert F.product_company_score(services) < 0.3
    assert F.product_company_score(product) > 0.9

def test_title_relevance_rewards_ml_titles():
    assert F.title_score("Machine Learning Engineer", []) > 0.8
    assert F.title_score("Marketing Manager", []) < 0.2

def test_location_score_targets_indian_hubs():
    assert F.location_score("Pune, Maharashtra", False) > 0.8
    assert F.location_score("London", False) < 0.4
    assert F.location_score("London", True) > F.location_score("London", False)

def test_education_tier_score():
    assert F.education_score([{"tier": "tier_1"}]) > F.education_score([{"tier": "tier_4"}])
    assert 0.0 <= F.education_score([]) <= 1.0
