"""
@edit-hist:
- fsm_lockin_s5
- fsm_s12_sleep
- fsm_4band_s5
- fsm_s4_dense_brake
- fsm_s5_nonlin_brake

@edit-desc:
- Monolith radial-only sensor
- Proposes a 2-mode distance-based FSM to drastically reduce the average sensing cost while perfectly holding the agent at the target. Mode 0 ("Active") uses `SectorRadar(s=5, d=(0.01,))` to quickly and reliably steer the point mass towards the origin. Once the agent is within the reward threshold (`r < 0.01`), it transitions to Mode 1 ("Hold"), which utilizes an ultra-low-cost `SectorRadar(s=1, d=(0.01,))`.
- Because the environment is velocity-controlled and commanding `v=0` eliminates all process noise, the agent won't experience drift. The policy easily learns to output `v=0` upon entering Mode 1, thereby perfectly locking the agent exactly at the origin for the remainder of the episode. This maximizes reward accumulation while spending the majority of the episode in an ultra-cheap sensor mode.
- Following the insights from the design log, this configuration optimizes the "Sprint" phase to lower the overall episode resource cost. The Active mode's sensor is upgraded to `SectorRadar(s=12, d=(0.01, 0.05, 0.1))`. The significantly higher angular resolution (`s=12`) enables the agent to navigate much more directly to the center, thereby decreasing the time it spends far from the origin.
- Furthermore, refined distance bands at `0.05` and `0.1` act as optimal braking markers. Given that the max step distance is `v_max * dt = 0.02` meters, these precise distance bins provide the RL policy with exactly the contextual awareness it needs to decelerate efficiently right before hitting the `0.01` threshold, preventing boundary overshoots.
- While the Active mode is slightly more expensive per-step, dropping its occupancy to well below 10% ensures that the agent transitions into the ultra-cheap `SectorRadar(s=1)` holding state quicker, resulting in a substantially lower average cost across the episode and higher overall task performance. The `initial_mode` is also updated to properly reflect the conditional boundaries.
- Following the insights from the design log, this update optimizes the Active mode by adopting a denser 4-band configuration (`d=(0.01, 0.02, 0.04, 0.08)`) combined with an angular resolution of `s=5`. By providing highly precise braking markers right up to the `0.01` threshold (where the max step size is `0.02`), the policy is equipped with the exact contextual awareness needed to rapidly approach and gracefully halt without overshooting. This is expected to minimize the "Sprint" time and reduce Active mode occupancy further.
- We retain the ultra-cheap two-way Sleep mode (`SectorRadar(s=1, d=(0.01,))`). While a completely blind Sleep mode `d=()` might seem mathematically cheaper, it strictly violates the observability constraint (as the `r >= 0.01` transition would no longer be decidable from the sensor observation) and empirically leads to unrecoverable drift. Thus, retaining the minimal distance band is essential for FSM reliability.
- Following the design log's hypothesis, we reduce the angular resolution of the Active mode from `s=5` to `s=4`. By retaining the highly effective 4-band braking configuration `d=(0.01, 0.02, 0.04, 0.08)`, we expect the dense radial context to sufficiently compensate for the reduced angular precision. This lowers the active per-step cost, which combined with the extremely cheap sleep mode, should reduce overall resource cost without significantly inflating approach time. We purposefully avoid making the Sleep mode "blind" (`d=()`) because doing so would violate the observability constraint, causing the monitor policy to fail at predicting the wake transition and leading to catastrophic drift.
- Following the insights from the design log, this update restores the optimal angular resolution of `s=5` for the Active mode, but refines the braking gradient to a broader, non-linear progression `d=(0.01, 0.03, 0.07, 0.15)`. This wider radial context aims to better capture the global layout and accelerate the early sprint phase, potentially pushing the active mode occupancy below 11% while maintaining precise enough markers for graceful braking near the target threshold.
"""
import jax.numpy as jnp
from lemca.tasks.p2d.base import DesignABC, GPS, SectorRadar


class Design(DesignABC):
    @property
    def mode_obs(self):
        ## EVOLVE-BLOCK-START
        return {
            0: SectorRadar(s=5, d=(0.01, 0.03, 0.07, 0.15)),
            1: SectorRadar(s=1, d=(0.01,))
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self):
        ## EVOLVE-BLOCK-START
        return [
            (0, 1, lambda state: jnp.linalg.norm(state) < 0.01),
            (1, 0, lambda state: jnp.linalg.norm(state) >= 0.01)
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        return lambda state: jnp.where(jnp.linalg.norm(state) < 0.01, 1, 0)
        ## EVOLVE-BLOCK-END
