"""Tests for modules.scam."""

from modules.scam import rule_score, scan_message, module_risk_score


def test_otp_message_detected_as_scam():
    r = scan_message("Dear customer, share your OTP now.")
    assert r.label == "scam"
    assert "otp_request" in r.signals


def test_kyc_message_detected():
    r = scan_message("Your KYC is pending. Update now at bit.ly/xyz.")
    assert r.label == "scam"
    assert "kyc_block" in r.signals


def test_prize_message_detected():
    r = scan_message("You have won Rs 500000! Click bit.ly/claim.")
    assert r.label == "scam"
    assert "prize_bait" in r.signals


def test_phishing_url_detected():
    r = scan_message("Verify your PIN at hbl-secure.tk/login immediately.")
    assert r.label == "scam"
    assert "phishing_link" in r.signals


def test_legitimate_message_not_flagged():
    r = scan_message("Your Careem ride is complete. Fare Rs 450.")
    assert r.label == "legitimate"
    assert r.score < 40


def test_legitimate_bill_not_flagged():
    r = scan_message("Your electricity bill for February is Rs 3450.")
    assert r.label == "legitimate"


def test_empty_message_handled():
    r = scan_message("")
    assert r.label == "legitimate"
    assert r.score == 0


def test_module_risk_score_uses_max():
    r1 = scan_message("Share OTP now.")
    r2 = scan_message("Your Careem ride is complete.")
    assert module_risk_score([r1, r2]) == r1.score


def test_rule_score_returns_tuple():
    score, signals = rule_score("Share your OTP now.")
    assert isinstance(score, int)
    assert isinstance(signals, list)