def fatigue_score(metrics, total_sets):

    fatigue = (
        metrics["intensity"] * 0.4 + metrics["error_rate"] * 100 * 0.3 + total_sets * 5
    )

    return min(100, round(fatigue, 2))
