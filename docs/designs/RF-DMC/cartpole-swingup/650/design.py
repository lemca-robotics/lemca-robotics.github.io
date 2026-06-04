"""
@edit-hist:
- monolith_n=60_Q=A_H=4
- relax_cart_q_e
- velocity_aware_H_reduction
- min_swingup_wide_cart

@edit-desc:
- monolithic design
- Transitions to an intermediate-quality (Q=E) balance mode with relaxed cart position bounds but tight pole angle bounds. This prevents benign cart drifts from triggering the expensive swing-up mode, aiming to dramatically increase the occupancy of the much cheaper balance mode while maintaining overall stability.
- Reduces the history length for both the swing-up (H=8) and balance (H=4) modes to reduce baseline costs. Also relaxes the spatial bounds for the balance mode and introduces velocity checks (`cart_velocity` and `pole_angular_velocity` < 1.0) into its entry condition, preventing premature transitions before the system is truly settled.
- Removes restrictive velocity checks from mode transitions, allowing the agent to enter the cheaper balance mode earlier. Widens the balance mode's cart exit bound to `> 1.75` to maximize occupancy of the cheaper mode, while reverting its history to the more stable `H=8`.
"""
import jax.numpy as jnp
from lemca.tasks.rs.base import DesignABC, RayScanSensor

class Design(DesignABC):
    @property
    def mode_obs(self):
        ## EVOLVE-BLOCK-START
        return {
            0: RayScanSensor(nrays=3, quality="D", history=8),
            1: RayScanSensor(nrays=3, quality="E", history=8)
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self):
        ## EVOLVE-BLOCK-START
        return [
            (0, 1, lambda state: (jnp.abs(state["pole_angle"]) < 0.2) & (jnp.abs(state["cart_position"]) < 1.2)),
            (1, 0, lambda state: (jnp.abs(state["pole_angle"]) > 0.25) | (jnp.abs(state["cart_position"]) > 1.75))
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        def f(mdp_state):
            cond = (jnp.abs(mdp_state["pole_angle"]) < 0.2) & (jnp.abs(mdp_state["cart_position"]) < 1.2)
            return jnp.where(cond, 1, 0)
        return f
        ## EVOLVE-BLOCK-END
