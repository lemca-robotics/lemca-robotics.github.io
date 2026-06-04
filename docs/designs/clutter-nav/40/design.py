"""
@name:
- cruise_evade_fsm_omnidirectional

@desc:
- Implements a dual-mode FSM distinguishing between a compute-efficient Cruise mode (`period=3`) and a responsive Evade mode (`period=1`). Maintains a full 360° field of view and maximum 15m radar range across both modes to ensure reliable trajectory scoring and prevent safety drops due to "unseen" dynamic threats. The transitions strictly utilize centroid distances and Time-To-Collision (TTC) to escalate priority conservatively.
"""
from typing import Any, Callable, Sequence, Tuple
from lead.tasks.nav.task import State
from lead.tasks.nav.base import Configuration, DesignABC, MonitorObservation

import numpy as np


class Design(DesignABC):
    @property
    def mode_obs(self) -> dict[int, Configuration]:
        ## EVOLVE-BLOCK-START
        return {
            0: Configuration(
                R0=8.0, max_range=15.0, fov=2 * np.pi,
                n_iters=1, n_samples=256, horizon=2.0, period=3
            ),
            1: Configuration(
                R0=8.0, max_range=15.0, fov=2 * np.pi,
                n_iters=1, n_samples=256, horizon=2.0, period=1
            )
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self) -> Sequence[Tuple[int, int, Callable[[MonitorObservation], bool]]]:
        ## EVOLVE-BLOCK-START
        def to_evade(obs: MonitorObservation) -> bool:
            if len(obs.obs.r) == 0:
                return False
            danger = (obs.obs.r < 2.5) | (
                (np.abs(obs.obs.az) < 2.1) & (obs.obs.v_r < -0.1) & (obs.obs.r < -2.7 * obs.obs.v_r)
            )
            return bool(np.any(danger))

        def to_cruise(obs: MonitorObservation) -> bool:
            if len(obs.obs.r) == 0:
                return True
            danger = (obs.obs.r < 2.8) | (
                (np.abs(obs.obs.az) < 2.2) & (obs.obs.v_r < -0.1) & (obs.obs.r < -3.0 * obs.obs.v_r)
            )
            return not bool(np.any(danger)) and obs.t_since_switch > 3

        return [
            (0, 1, to_evade),
            (1, 0, to_cruise)
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        def f(state: State) -> int:
            return 0

        return f
        ## EVOLVE-BLOCK-END
