"""
loan_offer.py
--------------
Real-world banking calculation layer for CreditSense AI.

Once a risk prediction is made, a real lender doesn't just say "Low Risk" --
they attach a concrete offer: an interest rate tier (driven by CIBIL score
and risk classification, as Indian banks/NBFCs actually price loans), and
the resulting EMI (Equated Monthly Installment), total interest, and total
payable amount over the loan tenure. This module implements that pricing
logic and the standard EMI formula.

This is a deterministic, rule-based banking calculation -- not a machine
learning model -- but it's what turns a bare risk label into something a
loan officer (or an applicant) can actually act on.
"""


def get_interest_rate(risk_label: str, cibil_score: float, collateral_provided: str):
    """Returns an annual interest rate (%) for the given risk profile, or
    None if the applicant is not eligible for standard financing.

    This mirrors how Indian lenders commonly tier interest rates: CIBIL
    score is the primary driver, High Risk applicants pay a higher premium
    (or are declined outright below a hard floor), and collateral earns a
    small rate discount (secured loans carry less lender risk).
    """
    collateral_discount = 0.5 if collateral_provided == "Yes" else 0.0

    if risk_label == "High Risk":
        if cibil_score < 500:
            return None  # not eligible for standard financing
        base_rate = 17.5 if cibil_score < 600 else 15.5
        return round(base_rate - collateral_discount, 2)

    # Low Risk tiers, by CIBIL band
    if cibil_score >= 800:
        base_rate = 8.5
    elif cibil_score >= 750:
        base_rate = 9.5
    elif cibil_score >= 700:
        base_rate = 10.5
    elif cibil_score >= 650:
        base_rate = 11.5
    else:
        base_rate = 13.0

    return round(base_rate - collateral_discount, 2)


def calculate_emi(principal: float, annual_rate_pct: float, tenure_months: int):
    """Standard EMI (Equated Monthly Installment) formula:

        EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)

    where P = principal, r = monthly interest rate (annual_rate / 12 / 100),
    n = number of monthly installments.

    Returns (monthly_emi, total_interest, total_payable), all rounded to
    2 decimal places.
    """
    if annual_rate_pct is None or principal <= 0 or tenure_months <= 0:
        return None, None, None

    monthly_rate = annual_rate_pct / 12 / 100

    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        factor = (1 + monthly_rate) ** tenure_months
        emi = principal * monthly_rate * factor / (factor - 1)

    total_payable = emi * tenure_months
    total_interest = total_payable - principal

    return round(emi, 2), round(total_interest, 2), round(total_payable, 2)


def build_loan_offer(risk_label: str, cibil_score: float, collateral_provided: str,
                      loan_amount: float, loan_duration: int):
    """Builds the full loan offer block attached to a prediction response.

    Returns a dict with eligibility, interest rate, EMI, and totals -- or
    an eligible=False result with a reason if the applicant doesn't
    qualify for standard financing.
    """
    rate = get_interest_rate(risk_label, cibil_score, collateral_provided)

    if rate is None:
        return {
            "eligible": False,
            "reason": "CIBIL score is below the minimum threshold for standard financing. "
                      "Refer to manual underwriting or a secured/collateral-backed product.",
        }

    emi, total_interest, total_payable = calculate_emi(loan_amount, rate, int(loan_duration))

    return {
        "eligible": True,
        "interest_rate_pct": rate,
        "monthly_emi": emi,
        "total_interest": total_interest,
        "total_payable": total_payable,
        "loan_amount": loan_amount,
        "tenure_months": int(loan_duration),
    }
