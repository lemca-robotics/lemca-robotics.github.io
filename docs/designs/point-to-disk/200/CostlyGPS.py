"""
@edit-hist:
- monorad_s4_n04
- monorad_s4_n08
- relaxed_fsm_s4
- relaxed_fsm_s3_d03

@edit-desc:
- Monolith radial-only sensor
- Based on the design log, monolithic `SectorRadar` strongly dominates `GPS` configurations and FSM variants for target performance < 400. To attain the target performance (>= 200) with minimal resource cost, we can use `s=4` sectors. Adding mild angular noise (`sigma_theta=0.4`) degrades the angular resolution slightly but still remains effective, thereby safely lowering the per-step cost close to the minimum while comfortably sustaining performance above the 200 threshold. Utilizing an FSM to toggle sensors based on distance is structurally counterproductive here, as the radar must be given distance bands (`d`) to evaluate the FSM predicates, incurring an additional `dist_cost` that offsets any mode-switching savings.
- The previous monolithic design with `SectorRadar(s=4, sigma_theta=0.4)` achieved an excellent task performance of ~351, massively overshooting the target of 200. To further compress the resource cost, we can aggressively increase the angular noise injection to `sigma_theta=0.8` (the maximum allowed menu value). This drops the angular resolution and the associated information cost while maintaining the fundamental 4-quadrant heading information that the point mass requires to navigate towards the origin.
- Implements a Minimalist Wake-Up FSM that transitions between an active `SectorRadar(s=4)` tracker and an ultra-low-power `SectorRadar(s=1)` monitor. To safely maximize the time spent in the cheap monitor mode while preserving `task_performance >= 200`, we employ the 'Relaxed Boundary Margins' strategy. The active tracker pushes the agent into the reward zone (`< 0.01`), but the monitor mode only wakes the tracker back up when the agent drifts past `dist >= 0.015`. This significantly extends the time the agent can "sleep" in the cheap monitor mode, sharply lowering the overall aggregate resource cost, while retaining enough reward zone occupancy to comfortably clear the 200 target threshold. FSM predicates are rigorously matched to crisp sensor distance bands (`sigma_r=0.0`) to maintain strict decidability.
- Aggressively relaxes the wake-up FSM threshold to `d=(0.03,)` to further maximize cheap monitor mode occupancy and lower aggregate resource cost. Replaces the `s=4` active tracker with the structurally minimal `s=3` (minimum geometry required for 2D heading) combined with maximum angular noise (`sigma_theta=0.8`) to establish a lower active tracking cost floor while comfortably maintaining task performance above the target threshold.
"""
import jax.numpy as jnp
from lemca.tasks.p2d.base import DesignABC, GPS, SectorRadar


class Design(DesignABC):
    @property
    def mode_obs(self):
        ## EVOLVE-BLOCK-START
        return {
            0: SectorRadar(s=3, d=(0.01,), sigma_theta=0.8, sigma_r=0.0),
            1: SectorRadar(s=1, d=(0.03,), sigma_r=0.0)
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self):
        ## EVOLVE-BLOCK-START
        return [
            (0, 1, lambda state: jnp.linalg.norm(state) < 0.01),
            (1, 0, lambda state: jnp.linalg.norm(state) >= 0.03),
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        return lambda _: 0
        ## EVOLVE-BLOCK-END

