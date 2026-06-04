"""
@edit-hist:
- gps_to_idle_fsm
- noisy_approach_hist_wide_hyst

@edit-desc:
- Monolith radial-only sensor
- Uses a 2-mode FSM to achieve high task performance while dramatically reducing average resource costs.
    - **Mode 0 (Approach)** utilizes a noiseless `GPS` for highly precise navigation towards the origin, effectively imitating the high performance of a monolithic `GPS(0.0)` controller.
    - **Mode 1 (Hover)** uses an ultra-cheap, noiseless `SectorRadar` equipped with a single distance threshold `d=(0.005,)` acting as a boundary detector.
- Once the agent enters the reward zone (`r < 0.01`) and safely crosses the `0.005` margin, it transitions to Mode 1. In this mode, the optimal policy maps the constant "inside boundary" observation to a zero velocity command. Since commanding zero velocity eliminates all process noise, the point mass cleanly stops and permanently "freezes" within the reward zone. This limits the expensive `GPS` cost to just the first few seconds of the approach phase, bringing the expected episode cost down considerably. Both transition predicates are strictly perfectly decidable from their respective source sensors without any monitor classification errors, fully satisfying the observability constraint.
- Updates the approach-and-hold FSM to use a highly cost-efficient noisy `GPS` (sigma_xy=0.01) with temporal filtering (`history=3`) in Mode 0, effectively reducing the dominant cost of the exact approach while smoothing out coarse navigation. To maintain observability of the inward transition boundary without error, a zero-cost 1-bin `SectorRadar(d=(0.008,))` is paired with the GPS. Additionally, we employ a wider hysteresis boundary (in at `0.008`, out at `0.0099`) to increase the stability of the stationary hover mode, helping the agent remain perfectly captured within the reward zone once it arrives.
"""
import jax.numpy as jnp
from lemca.tasks.p2d.base import DesignABC, GPS, SectorRadar


class Design(DesignABC):
    @property
    def mode_obs(self):
        ## EVOLVE-BLOCK-START
        return {
            0: (GPS(sigma_xy=0.01, history=3), SectorRadar(s=1, d=(0.008,), sigma_r=0.0, sigma_theta=0.0)),
            1: SectorRadar(s=1, d=(0.0099,), sigma_r=0.0, sigma_theta=0.0)
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self):
        ## EVOLVE-BLOCK-START
        return [
            (0, 1, lambda state: jnp.linalg.norm(state) < 0.008),
            (1, 0, lambda state: jnp.linalg.norm(state) >= 0.0099),
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        return lambda state: jnp.where(jnp.linalg.norm(state) < 0.008, 1, 0).astype(jnp.int32)
        ## EVOLVE-BLOCK-END
