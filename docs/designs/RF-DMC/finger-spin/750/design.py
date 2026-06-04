"""
@edit-hist:
- monolith_n=60_Q=A_H=4
- fsm_n8_d4_e2
- fsm_n8_d3_e2_early

@edit-desc:
- monolithic design
- Tests the open direction of shaving cost in the spin-up mode by degrading it to `Q=D`, while keeping the maintain mode at the previously identified optimal `Q=E, H=2`. Both modes maintain a consistent `n=8` spatial resolution to ensure FSM policy stability.
- Reduces the spin-up mode history from 4 to 3 to shave off base cost, and transitions into the cheaper maintain mode earlier (-17.0 rad/s with a fallback at -15.5 rad/s). This narrower buffer is designed to increase the occupancy of the low-cost maintain mode while keeping the speed safely above the 15 rad/s threshold.
"""
import jax.numpy as jnp
from lemca.tasks.rs.base import DesignABC, RayScanSensor

class Design(DesignABC):
    @property
    def mode_obs(self):
        ## EVOLVE-BLOCK-START
        return {
            0: RayScanSensor(nrays=8, quality="D", history=3),
            1: RayScanSensor(nrays=8, quality="E", history=2),
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self):
        ## EVOLVE-BLOCK-START
        return [
            (0, 1, lambda state: state["hinge_angvel"] < -17.0),
            (1, 0, lambda state: state["hinge_angvel"] > -15.5),
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        def f(mdp_state):
            return 0
        return f
        ## EVOLVE-BLOCK-END
