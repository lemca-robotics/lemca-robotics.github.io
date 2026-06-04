"""
@edit-hist:
- monolith_n=180_Q=B_H=4
- linf_fsm_sparse_catch
- kinematic_hold_catch

@edit-desc:
- monolithic design
- Introduces a 3-mode L-infinity distance FSM to segregate the controller into "Swing", "Catch", and "Hold" phases. The critical "Catch" phase is assigned a high-quality sparse sensor (`n=12`, `Q="B"`) providing clean signal without excessive cost, while "Swing" and "Hold" utilize cheap, low-resolution sensors to drastically minimize overall resource consumption. The Catch distance threshold is widened to 0.25 to provide the control policy more reaction time.
- Implements a "Pure Kinematic Hold" by completely removing the `RayScanSensor` during the Hold phase (Mode 2), relying entirely on the base `TaskState` kinematic features to balance the ball. This perfectly leverages the >90% occupancy of Mode 2 to drastically cut the overall resource cost. The `initial_mode` logic is also upgraded to accurately place the system in the correct FSM mode from step 0.
"""
import jax.numpy as jnp
from lemca.tasks.rs.base import DesignABC, RayScanSensor

class Design(DesignABC):
    @property
    def mode_obs(self):
        ## EVOLVE-BLOCK-START
        return {
            0: RayScanSensor(nrays=4, quality="C", history=4),
            1: RayScanSensor(nrays=12, quality="B", history=4),
            2: (),
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self):
        ## EVOLVE-BLOCK-START
        def m0_to_m2(state):
            return jnp.max(jnp.abs(state["ball_to_target"])) < 0.04

        def m0_to_m1(state):
            return jnp.max(jnp.abs(state["ball_to_target"])) < 0.25

        def m1_to_m2(state):
            return jnp.max(jnp.abs(state["ball_to_target"])) < 0.04

        def m1_to_m0(state):
            return jnp.max(jnp.abs(state["ball_to_target"])) >= 0.25

        def m2_to_m1(state):
            dist = jnp.max(jnp.abs(state["ball_to_target"]))
            return jnp.logical_and(dist >= 0.06, dist < 0.25)

        def m2_to_m0(state):
            return jnp.max(jnp.abs(state["ball_to_target"])) >= 0.25

        return [
            (0, 2, m0_to_m2),
            (0, 1, m0_to_m1),
            (1, 2, m1_to_m2),
            (1, 0, m1_to_m0),
            (2, 1, m2_to_m1),
            (2, 0, m2_to_m0),
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        def f(mdp_state):
            dist = jnp.max(jnp.abs(mdp_state["ball_to_target"]), axis=-1)
            return jnp.where(dist < 0.04, 2, jnp.where(dist < 0.25, 1, 0)).astype(jnp.int32)
        return f
        ## EVOLVE-BLOCK-END
