"""
Performance Metrics Engine
--------------------------------
Calculates:

- Accuracy
- Precision (form consistency)
- Intensity
- Efficiency
- Final Score
"""


def calculate_metrics(session):

    total_reps = session.get("total_reps", 0)
    duration = session.get("duration", 1)
    errors = session.get("errors", {})

    total_errors = sum(errors.values())

    # ---------------------------------
    # 1️⃣ ERROR RATE (normalized per-rep)
    # Errors in the tracker are often counted per-frame; normalize by
    # reps and clamp to a reasonable cap so a large raw frame count
    # doesn't zero-out accuracy unhelpfully.
    # ---------------------------------
    avg_errors_per_rep = total_errors / total_reps if total_reps > 0 else 0

    # Empirical cap: treat >=5 error-events per rep as a full error.
    errors_per_rep_cap = 5.0
    error_rate = min(avg_errors_per_rep / errors_per_rep_cap, 1.0)

    # ---------------------------------
    # 2️⃣ ACCURACY (Form Quality)
    # ---------------------------------
    accuracy = max(0.0, (1.0 - error_rate) * 100.0)

    # ---------------------------------
    # 3️⃣ PRECISION (Consistency)
    # Penalizes uneven error distribution across error types. Normalize
    # worst-case errors per rep using same cap to keep metrics stable.
    # ---------------------------------
    if total_errors > 0:
        worst_error = max(errors.values())
        worst_per_rep = worst_error / total_reps if total_reps > 0 else worst_error
        precision = max(0.0, 100.0 - (min(worst_per_rep / errors_per_rep_cap, 1.0) * 100.0))
    else:
        precision = 100.0

    # ---------------------------------
    # 4️⃣ INTENSITY (Reps per Minute)
    # ---------------------------------
    intensity = total_reps / (duration / 60) if duration > 0 else 0

    # ---------------------------------
    # 5️⃣ EFFICIENCY
    # Ideal rep time = 3 sec
    # ---------------------------------
    ideal_duration = total_reps * 3
    efficiency = (
        (ideal_duration / duration) * 100 if duration > 0 else 0
    )

    efficiency = min(efficiency, 120)

    # ---------------------------------
    # 6️⃣ FINAL SCORE
    # ---------------------------------
    final_score = (
        accuracy * 0.4
        + precision * 0.2
        + min(efficiency, 100) * 0.2
        + min(intensity * 4, 100) * 0.2
    )

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "intensity": round(intensity, 2),
        "efficiency": round(efficiency, 2),
        "error_rate": round(error_rate, 2),
        "score": round(final_score, 2),
    }
