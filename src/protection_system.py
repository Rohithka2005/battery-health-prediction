"""Rule-based battery status recommendations for predicted RUL."""
import math


def classify_battery(predicted_rul: float) -> tuple[str, str]:
    """Return project-defined status and suggested action for a predicted RUL."""
    try:
        rul = float(predicted_rul)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predicted RUL must be a finite number.") from exc

    if not math.isfinite(rul):
        raise ValueError("Predicted RUL must be a finite number.")
    if rul < 0:
        raise ValueError("Predicted RUL cannot be negative.")

    if rul > 100:
        return "HEALTHY", "Continue normal operation"
    if rul > 40:
        return "MODERATE DEGRADATION", "Enable power-saving mode"
    return "CRITICAL", "Reduce load and schedule maintenance"
