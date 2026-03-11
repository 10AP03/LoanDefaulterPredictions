const form = document.getElementById("loanForm");
const resultBox = document.getElementById("result");
const submitBtn = document.getElementById("submitBtn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Clear old result
  resultBox.className = "hidden";
  resultBox.innerHTML = "";

  // Loading state so user can't spam click
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  // Collect form data
  const payload = {
    Age: Number(document.getElementById("Age").value),
    Income: Number(document.getElementById("Income").value),
    LoanAmount: Number(document.getElementById("LoanAmount").value),
    CreditScore: Number(document.getElementById("CreditScore").value),
    MonthsEmployed: Number(document.getElementById("MonthsEmployed").value),
    NumCreditLines: Number(document.getElementById("NumCreditLines").value),
    InterestRate: Number(document.getElementById("InterestRate").value),
    LoanTerm: Number(document.getElementById("LoanTerm").value),
    DTIRatio: Number(document.getElementById("DTIRatio").value),
    HasMortgage: Number(document.getElementById("HasMortgage").value),
    HasDependents: Number(document.getElementById("HasDependents").value)
  };

  try {
    // hanged URL to relative "/predict"
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.message || "Prediction failed");
    }

    // Decide risk color
    let riskClass = "";
    if (result.risk_level === "LOW") riskClass = "low";
    if (result.risk_level === "MEDIUM") riskClass = "medium";
    if (result.risk_level === "HIGH") riskClass = "high";

    resultBox.className = riskClass;
    resultBox.innerHTML = `
      <h3>Prediction Result</h3>
      <p><b>Default Probability:</b> ${result.default_probability}</p>
      <p><b>Risk Level:</b> ${result.risk_level}</p>
      <p><b>Threshold Used:</b> ${result.threshold}</p>
    `;

  } catch (error) {
    resultBox.className = "high";
    resultBox.innerHTML = `
      <h3>Error</h3>
      <p>${error.message}</p>
    `;
  }

  // Re-enable button
  submitBtn.disabled = false;
  submitBtn.textContent = "Predict Risk";

  resultBox.classList.remove("hidden");
});