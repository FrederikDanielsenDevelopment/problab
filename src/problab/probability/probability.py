#P(event, method="auto")
#P(event, method="exact")
#P(event, method="monte_carlo")
import numpy as np

from src.problab.events import Event
from src.problab.probability._config import DEFAULT_PROB_NUM_SAMPLES
from src.problab.probability.results import ProbabilityResult
from src.problab.random_variables.context import RealizationContext


def P(event: Event,
      given: Event | None = None,
      num_samples: int = DEFAULT_PROB_NUM_SAMPLES,
      rng: np.random.Generator | None = None
    ) -> ProbabilityResult:

    if not isinstance(event, Event):
        raise TypeError("'event' must be a Event.")

    if given is not None and not isinstance(given, Event):
        raise TypeError("'given' must be an Event or None.")

    if not isinstance(num_samples, int):
        raise TypeError("'num_samples' must be an integer.")

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("'rng' must be a np.random.Generator or None.")

    if num_samples <= 0:
        raise ValueError("'num_samples' must be positive.")

    context = RealizationContext(num_samples=num_samples, rng=rng)

    event_values = context.evaluate(event._node)

    if given is None:
        return ProbabilityResult(
            value=float(np.mean(event_values)),
            num_successes=int(np.sum(event_values)),
            num_unconditioned_samples=num_samples
        )

    given_values = context.evaluate(given._node)
    num_conditioned_samples  = int(np.sum(given_values))

    if num_conditioned_samples == 0:
        return ProbabilityResult(
            value=np.nan,
            num_successes=0,
            num_unconditioned_samples=num_samples,
            num_conditioned_samples=0,
        )

    conditioned_event_values = event_values[given_values]

    return ProbabilityResult(
        value=float(np.mean(conditioned_event_values)),
        num_successes=int(np.sum(conditioned_event_values)),
        num_unconditioned_samples=num_samples,
        num_conditioned_samples=num_conditioned_samples,
    )



