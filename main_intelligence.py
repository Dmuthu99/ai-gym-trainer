from workout_tracker.bicep.bicep_curl import run_bicep_session
from intelligence.analysis.performance_engine import calculate_metrics
from intelligence.prediction.fatigue_model import fatigue_score
from intelligence.prediction.injury_model import injury_risk

print("Starting Bicep Workout...")

session = run_bicep_session()

print("\nSession Data:")
print(session)

metrics = calculate_metrics(session)

fatigue = fatigue_score(metrics, session["total_sets"])
risk = injury_risk(fatigue, metrics["error_rate"])

print("\n--- AI ANALYSIS ---")
print("Accuracy:", metrics["accuracy"], "%")
print("Efficiency:", metrics["efficiency"], "%")
print("Intensity:", metrics["intensity"])
print("Overall Score:", metrics["score"])
print("Fatigue Level:", fatigue)
print("Injury Risk Probability:", risk, "%")
