"""Step predicates for the paper's synchronized HaLoRA alignment schedule."""


def should_refresh_noise(global_step: int, refresh_steps: int) -> bool:
    if global_step < 0 or refresh_steps <= 0:
        raise ValueError("global_step must be non-negative and refresh_steps positive")
    return global_step % refresh_steps == 0


def should_apply_regularizer(global_step: int, interval: int) -> bool:
    if global_step < 0 or interval <= 0:
        raise ValueError("global_step must be non-negative and interval positive")
    return global_step % interval == 0
