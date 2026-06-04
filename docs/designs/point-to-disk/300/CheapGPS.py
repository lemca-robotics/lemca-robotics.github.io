"""
@edit-hist:
- noiseless_approach_blind_station
- three_tier_precision_fsm
- one_way_precision_cascade
- four_tier_tuned_cascade

@edit-desc:
- Monolith radial-only sensor
- Implemented the breakthrough dual-mode strategy from the design log.
- Mode 0 uses a perfectly noiseless GPS (`sigma_xy=0.0`) for a highly precise approach to the reward zone. The zero-noise sensor ensures the monitor policy can flawlessly trigger the `< 0.01` transition boundary.
- Once inside the reward zone, the agent switches to Mode 1, which utilizes an ultra-cheap, angularly-blind `SectorRadar(s=1, d=(0.01,))` for station-keeping. By commanding `v=0` in this blind state, the agent entirely eliminates action-dependent process noise, safely locking itself inside the target indefinitely. The FSM falls back to Mode 0 if the agent ever drifts out of the `0.01` radius, leveraging the radar's exact distance threshold. This temporal amortization drastically reduces expected resource costs while achieving optimal task performance.
- This design introduces a three-tier precision FSM structure based on the optimal strategies outlined in the design log. It seeks to minimize the time spent in the expensive precise GPS mode by reserving it exclusively for the final approach.
    - Mode 0 uses a coarse and cheaper `GPS(sigma_xy=0.05)` to rapidly navigate the point mass from its random initial spawn into a `0.1` radius of the origin.
    - Mode 1 switches to a precise `GPS(sigma_xy=0.01)` to accurately guide the agent into the exact `0.01` reward zone.
    - Mode 2 acts as a terminal idle state, using an ultra-cheap, angularly-blind `SectorRadar(s=1, d=(0.01,))` to lock the agent at the origin with zero velocity, completely eliminating process noise.
- Both active navigation modes (0 and 1) are fused with the zero-cost boolean boundary detector `SectorRadar(d=(0.01, 0.1))` to guarantee that FSM state boundaries are perfectly observable. This dramatically reduces per-step resource costs while maintaining exceptional task performance.
- Implements the 3-Tier Precision Cascade with a "one-way" FSM structure as identified in the design log.
    - Mode 0 uses a coarse and cheaper `GPS(sigma_xy=0.1)` to guide the agent to within `0.2` of the origin. It fuses this with a `SectorRadar(d=(0.2,))` to effectively observe the FSM state boundary.
    - Mode 1 takes over for the near approach (`< 0.2`) using a highly precise `GPS(sigma_xy=0.01)` paired with `SectorRadar(d=(0.01,))`. Crucially, omitting the backward transition `(Mode 1 -> Mode 0)` strips the need for the outer `0.2` distance band in Mode 1, minimizing sensor footprint.
    - Mode 2 remains an ultra-cheap idle mode for station-keeping using only the blind radar.
- This configuration achieves an exceptional pareto-optimal balance between process acquisition speed and per-step sensing costs.
- Upgrades the FSM to a 4-Tier Precision Cascade, directly addressing the SOTA configurations from the design log. This tailors the GPS precision optimally across four stages of approach to slash per-step sensing costs. We specifically expand the cascade boundaries to `0.4` and `0.2` to balance the poor SNR of the ultra-coarse `GPS(0.5)` during the outer approach, ensuring swift hand-offs and reducing overall time spent wandering.
"""
import jax.numpy as jnp
from lemca.tasks.p2d.base import DesignABC, GPS, SectorRadar


class Design(DesignABC):
    @property
    def mode_obs(self):
        ## EVOLVE-BLOCK-START
        return {
            0: (GPS(sigma_xy=0.5), SectorRadar(s=1, d=(0.4,))),
            1: (GPS(sigma_xy=0.1), SectorRadar(s=1, d=(0.2,))),
            2: (GPS(sigma_xy=0.01), SectorRadar(s=1, d=(0.01,))),
            3: SectorRadar(s=1, d=(0.01,))
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self):
        ## EVOLVE-BLOCK-START
        return [
            (0, 1, lambda state: jnp.linalg.norm(state) < 0.4),
            (1, 2, lambda state: jnp.linalg.norm(state) < 0.2),
            (2, 3, lambda state: jnp.linalg.norm(state) < 0.01),
            (3, 2, lambda state: jnp.linalg.norm(state) >= 0.01)
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        return lambda state: jnp.where(
            jnp.linalg.norm(state) < 0.01, 3,
            jnp.where(jnp.linalg.norm(state) < 0.2, 2,
            jnp.where(jnp.linalg.norm(state) < 0.4, 1, 0))
        )
        ## EVOLVE-BLOCK-END
