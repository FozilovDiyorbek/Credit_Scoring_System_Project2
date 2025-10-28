from pydantic import BaseModel
from enum import Enum

class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanIntent(str, Enum):
    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"


class CreditData(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: HomeOwnership
    person_emp_length: float
    loan_intent: LoanIntent
    loan_amnt: float
    loan_int_rate: float
    cb_person_cred_hist_length: float