"""
@edit-hist:
- fsm_gps_noise_adaptive
- fsm_optimal_station_keeping
- fsm_3mode_transit
- fsm_aggressive_hysteresis

@edit-desc:
- Monolith radial-only sensor
- Based on the design log and the Pareto trade-offs observed, GPS heavily outperforms Sector Radar. However, utilizing a highly precise GPS across the entire episode incurs unnecessary costs, especially during the initial transit phase to the origin. This design introduces a two-mode Finite State Machine (FSM):
    - 1. **Mode 0 (Far)**: Uses a very cheap, noisy GPS (`sigma_xy=0.1, history=2`) when the agent is far from the target. It provides enough precision to steer the agent broadly toward the origin at a lower step cost.
    - 2. **Mode 1 (Close)**: Transitions to a slightly more precise GPS (`sigma_xy=0.05, history=4`) when the agent approaches the origin (`dist < 0.2`). The extra precision and history compensate for process noise, allowing the agent to settle comfortably inside the strict 0.01 reward threshold while ensuring the task performance exceeds 200.
    - The transition predicates incorporate a margin (`0.2` inward, `0.3` outward hysteresis) to ensure that the monitor policy can reliably evaluate the transition from the noisy source observation space.
- A highly optimized 2-mode FSM that heavily minimizes both navigation and station-keeping costs.
    - Mode 0 handles navigation by pairing a highly precise GPS (`sigma_xy=0.01`) with a 1-sector radar (`d=(0.008,)`). The GPS ensures a fast, direct path to the origin (minimizing the expensive Mode 0 occupancy), while the radar perfectly resolves the `<0.008` transition boundary for the monitor policy.
    - Once inside, Mode 1 takes over using an ultra-cheap 1-sector radar (`d=(0.01,)`) just at the edge of the reward zone. This forces the agent to confidently output `v=0` to stay inside the boundary, completely eliminating process noise and collapsing the station-keeping cost to a mere ~0.028/step. The transition boundary is relaxed compared to previous designs (`0.008` vs `0.005`) to help the agent enter Mode 1 much faster, further minimizing total resource cost.
- Introduce a 3-mode FSM to decouple the task into three distinct phases:
    - 1. **Far Transit (Mode 0)**: Uses a cheap, noisy GPS (`sigma_xy=0.1, history=2`) to roughly navigate the agent to within `0.2` of the target.
    - 2. **Precision Approach (Mode 1)**: Uses a high-precision `GPS(0.01)` coupled with a 1-sector Radar to funnel the agent from `0.2` exactly into the strict `<0.008` reward zone.
    - 3. **Station Keeping (Mode 2)**: Uses an ultra-cheap 1-sector Radar (`d=(0.01,)`) to coast silently at `v=0`, eliminating process noise while monitoring for drift out of the reward boundary.
    - This multi-stage funnel minimizes expensive high-precision sensor occupancy by only using it for the final approach, further driving down resource costs while comfortably exceeding the target performance.
- Expanding the Mode 2 outward hysteresis boundary from `0.01` to `0.018`. This aggressively trades excess task performance (which is currently well above the 200 target) to significantly reduce the frequency of expensive active corrections. By allowing the agent to drift further out before triggering a switch back to the expensive Mode 1 precision approach, we decrease the occupancy of Mode 1, minimizing the total resource cost of the 3-mode transit funnel even further.
"""
import jax.numpy as jnp
from lemca.tasks.p2d.base import DesignABC, GPS, SectorRadar


class Design(DesignABC):
    @property
    def mode_obs(self):
        ## EVOLVE-BLOCK-START
        return {
            0: GPS(sigma_xy=0.1, history=2),
            1: (GPS(sigma_xy=0.01, history=1), SectorRadar(s=1, d=(0.008,), sigma_r=0.0)),
            2: SectorRadar(s=1, d=(0.018,), sigma_r=0.0)
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self):
        ## EVOLVE-BLOCK-START
        return [
            (0, 1, lambda state: jnp.linalg.norm(state) < 0.2),
            (1, 2, lambda state: jnp.linalg.norm(state) < 0.008),
            (1, 0, lambda state: jnp.linalg.norm(state) >= 0.3),
            (2, 1, lambda state: jnp.linalg.norm(state) >= 0.018)
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        return lambda state: jnp.where(
            jnp.linalg.norm(state) < 0.008, 2,
            jnp.where(jnp.linalg.norm(state) < 0.2, 1, 0)
        )
        ## EVOLVE-BLOCK-END
