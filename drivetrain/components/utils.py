"""Shared helpers for drivetrain component modules."""


def scaling_factor(name, factors):
    """Return an exact scaling factor, defaulting to 1.0 when absent."""
    return factors.get(name, 1.0)
