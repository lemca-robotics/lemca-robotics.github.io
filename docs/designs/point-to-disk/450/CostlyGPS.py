"""
@edit-hist:
- blind_hold_fsm
- aggressive_hysteresis_fsm
- optimized_depth_hysteresis

@edit-desc:
- Monolith radial-only sensor
- The proposed design introduces a highly efficient 2-mode distance-based FSM. Mode 0 is active in the far-field and uses `SectorRadar(s=16, d=(0.005, 0.1))` to steer the agent towards the origin. Once the agent approaches within a `0.005` radius—safely inside the strict `0.01` reward threshold—it transitions to Mode 1.
- Mode 1 acts as a "blind holding" mode using a minimalist `SectorRadar(s=1, d=(0.005,))`. By intentionally depriving the policy of angular position feedback near the origin, the agent naturally learns to output a constant zero-velocity action. Commanding zero velocity eliminates process noise (which scales with action magnitude), allowing the agent to remain perfectly still and accumulate reward indefinitely at near-zero sensor cost.
- By setting the FSM boundary to `0.005` rather than `0.01`, we create a robust buffer. If the agent experiences infinitesimal drifts that push it beyond `0.005`, it instantly transitions back to Mode 0 for correction, while still remaining inside the `0.01` reward zone. This decoupling ensures continuous task reward while slashing expected resource costs down to `< 1.0`.
- Expands the FSM hysteresis bounds to `< 0.006` (entry) and `>= 0.0095` (exit). By enlarging the holding mode's activation threshold, the agent transitions to the ultra-cheap `SectorRadar(s=1)` holding mode earlier and stays in it longer before triggering a correction. The exit threshold of `0.0095` provides a tight but robust `0.0005` safety margin inside the true `0.01` reward boundary, preventing dropped reward frames while driving the FSM into maximizing the holding mode occupancy and further lowering the episodic resource cost.
- Refines the hysteresis loop and approach sensor to maximize holding mode occupancy while preserving task performance. First, the FSM entry threshold is tightened from `< 0.006` to `< 0.005`. By driving the agent deeper into the origin before switching to the "blind" holding mode, the natural drift outward takes longer, significantly increasing the time spent in the ultra-cheap Mode 1. Second, the approach sensor is upgraded to include intermediate depth boundaries `d=(0.005, 0.05, 0.2)`. These extra bins allow the policy to smoothly decelerate, speeding up the overall transition into the holding mode and further driving down the episodic resource cost.
"""
import jax.numpy as jnp
from lemca.tasks.p2d.base import DesignABC, GPS, SectorRadar


class Design(DesignABC):
    @property
    def mode_obs(self):
        ## EVOLVE-BLOCK-START
        return {
            0: SectorRadar(s=16, d=(0.005, 0.05, 0.2)),
            1: SectorRadar(s=1, d=(0.0095,))
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self):
        ## EVOLVE-BLOCK-START
        return [
            (0, 1, lambda x: jnp.linalg.norm(x) < 0.005),
            (1, 0, lambda x: jnp.linalg.norm(x) >= 0.0095)
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        return lambda _: 0
        ## EVOLVE-BLOCK-END
