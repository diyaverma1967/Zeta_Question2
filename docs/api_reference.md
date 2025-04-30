# API Reference

## POST /loan/eligibility

**Request Body**:
```json
{
  "ApplicantIncome": 12000,
  "CoapplicantIncome": 3000,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": 0,
  "Education": "Graduate",
  "Self_Employed": "No",
  "Credit_History": 1,
  "Property_Area": "Urban"
}