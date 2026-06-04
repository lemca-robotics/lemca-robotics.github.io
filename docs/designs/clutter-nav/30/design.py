"""
@name:
- kinematic_fsm_asym
- alert_window_expansion
- alert_horizon_panic_tiering
- fov_restriction_tuned_fsm
- alert_centric_panic_deferral

@desc:
- Introduces a Kinematic/Directional FSM that scales clearance thresholds based on obstacle bearing (front vs rear) and approach velocity, significantly dropping Evasive mode occupancy. Pairs this with asymmetric MPPI configurations: a cheap `period=4, n_samples=256` Cruise mode and a highly-reactive `period=1, n_samples=512` Evasive mode.
- Introduces a 3-mode FSM that drastically widens the temporal and spatial buffer of a mid-tier Alert mode (`period=2`). By detecting approaching threats much earlier using a continuous radial velocity scaling metric, the FSM provides the cheaper Alert mode sufficient replan cycles to resolve dynamic encounters, thereby slashing the occupancy of the highly expensive Panic mode (`n_samples=512`). Reduces `max_range` in the evasive modes to further drop baseline coverage costs.
- Extends the Alert mode's planning horizon to 3.0s to find smoother avoidance paths, drastically reducing the need to enter Panic mode. Tiers the sensor `max_range` downwards (10.0m -> 8.0m -> 6.0m) to trim coverage costs in high-frequency modes, and tightens Panic transition thresholds slightly to prevent premature escalation from distant fast-approaching threats.
- Restricts the MPPI field-of-view across all modes to `1.5 * pi` to proportionally cut coverage costs while remaining kinematically safe, as the unicycle agent primarily responds to frontal threats. Pushes the Cruise mode's period to 15 to nearly eliminate baseline monitoring costs, extends the Alert mode's period to 3, and tightens the Panic clearance thresholds slightly to force the agent to resolve more encounters in the cheaper Alert mode.
- Drastically reduces resource cost by delegating more evasive maneuvers to the cheaper Alert mode, allowing Panic occupancy to fall. Widens the Alert mode's trigger thresholds while slightly relaxing the Panic mode's thresholds, reducing unnecessary escalations. Extends the Alert mode's period to 4 to further slash its compute footprint.
"""
from typing import Any, Callable, Sequence, Tuple
from lemca.tasks.nav.task import State
from lemca.tasks.nav.base import Configuration, DesignABC, MonitorObservation

import numpy as np


class Design(DesignABC):
    @property
    def mode_obs(self) -> dict[int, Configuration]:
        ## EVOLVE-BLOCK-START
        return {
            0: Configuration(
                R0=8.0, max_range=10.0, fov=1.5 * np.pi,
                n_iters=1, n_samples=256, horizon=2.0, period=15
            ),
            1: Configuration(
                R0=8.0, max_range=8.0, fov=1.5 * np.pi,
                n_iters=1, n_samples=256, horizon=3.0, period=4
            ),
            2: Configuration(
                R0=8.0, max_range=6.0, fov=1.5 * np.pi,
                n_iters=1, n_samples=512, horizon=2.0, period=1
            )
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self) -> Sequence[Tuple[int, int, Callable[[MonitorObservation], bool]]]:
        ## EVOLVE-BLOCK-START
        def to_panic(obs: MonitorObservation) -> bool:
            if len(obs.obs.r) == 0:
                return False
            clearance = obs.obs.r - obs.obs.sz - 0.5
            v_app = np.clip(obs.obs.v_r, -5.0, 0.0)
            eff_clearance = clearance + v_app * 0.5
            thresh = np.where(np.abs(obs.obs.az) < (np.pi / 2), 0.45, 0.2)
            return bool(np.any(eff_clearance < thresh))

        def to_alert(obs: MonitorObservation) -> bool:
            if len(obs.obs.r) == 0:
                return False
            clearance = obs.obs.r - obs.obs.sz - 0.5
            v_app = np.clip(obs.obs.v_r, -5.0, 0.0)
            eff_clearance = clearance + v_app * 1.0
            thresh = np.where(np.abs(obs.obs.az) < (np.pi / 2), 2.6, 1.4)
            return bool(np.any(eff_clearance < thresh))

        def to_cruise(obs: MonitorObservation) -> bool:
            if len(obs.obs.r) == 0:
                return True
            clearance = obs.obs.r - obs.obs.sz - 0.5
            v_app = np.clip(obs.obs.v_r, -5.0, 0.0)
            eff_clearance = clearance + v_app * 1.0
            thresh = np.where(np.abs(obs.obs.az) < (np.pi / 2), 3.0, 1.8)
            return bool(np.all(eff_clearance >= thresh))

        def safe_from_panic(obs: MonitorObservation) -> bool:
            if len(obs.obs.r) == 0:
                return True
            clearance = obs.obs.r - obs.obs.sz - 0.5
            v_app = np.clip(obs.obs.v_r, -5.0, 0.0)
            eff_clearance = clearance + v_app * 0.5
            thresh = np.where(np.abs(obs.obs.az) < (np.pi / 2), 0.75, 0.4)
            return bool(np.all(eff_clearance >= thresh))

        return [
            (0, 2, to_panic),
            (0, 1, to_alert),
            (1, 2, to_panic),
            (1, 0, to_cruise),
            (2, 0, to_cruise),
            (2, 1, safe_from_panic)
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        def f(state: State) -> int:
            return 0

        return f
        ## EVOLVE-BLOCK-END
