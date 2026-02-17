from intelligence.analysis.performance_engine import calculate_metrics

# Obtain session data. Prefer live run but fall back to sample data to
# avoid requiring a webcam during automated runs.
try:
	from workout_tracker.bicep.bicep_curl import run_bicep_session

	session = run_bicep_session()
except Exception:
	session = {
		"exercise": "bicep_curl",
		"total_sets": 3,
		"total_reps": 36,
		"duration": 300,
		"sets": [
			{"set": 1, "reps": 12},
			{"set": 2, "reps": 12},
			{"set": 3, "reps": 12},
		],
		"errors": {
			"back_not_straight": 2,
			"shoulder_not_level": 1,
			"hands_not_synced": 0,
			"arm_not_straight": 1,
		},
	}

metrics = calculate_metrics(session)

print("\n--- PERFORMANCE METRICS ---")
print("Accuracy:", metrics["accuracy"], "%")
print("Precision:", metrics["precision"], "%")
print("Efficiency:", metrics["efficiency"], "%")
print("Intensity:", metrics["intensity"], "reps/min")
print("Final Score:", metrics["score"]) 
