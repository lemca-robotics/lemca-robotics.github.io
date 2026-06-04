"""
@name:
- cruise_evade_fsm
- clearance_spatial_fsm
- lean_focused_evasion
- ttc_evasion_fsm
- lean_cruise_expanded_evasion
- tri_mode_escalation

@desc:
- Introduces a two-mode controller to significantly lower resource costs while maintaining high task performance. A low-cost "Cruise" mode (R0=12, period=2) is used for general navigation, and a high-resolution, high-frequency "Evasion" mode (R0=45, period=1) is triggered only when obstacles get dangerously close, specifically focusing on a larger safety buffer for frontal and side approaches.
- Transitions the mode-switching triggers from absolute range (`r`) to obstacle clearance (`r - sz`), providing robustness to varying obstacle sizes as proven by the probed designs. Retains the spatial filtering (a wide front cone) to aggressively reduce the occupancy of the expensive evasion mode. The evasion mode's configurations are significantly leaned out (`R0=24`, `max_range=12`) combined with a bump in `n_samples=512` to dramatically drop the base cost while ensuring an ultra-safe high-resolution MPPI plan when evasive maneuvers are triggered.
- Optimizes the dual-mode FSM to dramatically reduce the occupancy of the expensive evasion mode. Refines the spatial filter to a narrower 180° frontal cone (`np.pi / 2`) and tightens the clearance bounds so Evasion is only triggered for imminent threats. To guarantee safety despite the delayed reaction, Evasion's MPPI solver is beefed up (`n_iters=2`), while `max_range` is capped at 10m for both modes to strictly align with the 2.0s horizon and shave off unnecessary coverage costs.
- Transitions the mode-switching logic to use Time-to-Collision (TTC) alongside spatial clearance, significantly reducing the occupancy of the expensive evasion mode by avoiding premature activation for slow-moving distant threats. Lowers the radar power in Evasion (`R0=22`) and extends the Cruise mode's period (`period=3`) to drastically decrease the overall resource cost while retaining a potent high-resolution MPPI plan (`n_iters=2`) when evasion is genuinely required.
- Aggressively leans out the Cruise mode by extending its planning period to 4 steps and substantially cuts Evasion mode's radar power (R0=19) to optimize per-step costs. To compensate for the slightly delayed reaction and higher sensor noise, the spatial and TTC trigger boundaries are proactively expanded to guarantee robust obstacle avoidance while maintaining lower resource consumption.
- Evolves the architecture to a tri-mode FSM by introducing a mid-tier "Alert" mode (10Hz, R0=15) to bridge the gap between Cruise and Evasion. This escalation shifts the vast majority of avoidance handling to the significantly cheaper Alert mode, reserving the expensive 20Hz Evasion mode exclusively for immediate-proximity emergencies (e.g., front clearance < 2.1m or TTC < 1.4s), drastically reducing average cost without compromising safety.
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
                R0=10, max_range=10, fov=2*np.pi,
                n_iters=1, n_samples=256, horizon=2.0, period=4
            ),
            1: Configuration(
                R0=15, max_range=10, fov=2*np.pi,
                n_iters=1, n_samples=512, horizon=2.0, period=2
            ),
            2: Configuration(
                R0=19, max_range=10, fov=2*np.pi,
                n_iters=2, n_samples=512, horizon=2.0, period=1
            )
        }
        ## EVOLVE-BLOCK-END

    @property
    def mode_transitions(self) -> Sequence[Tuple[int, int, Callable[[MonitorObservation], bool]]]:
        ## EVOLVE-BLOCK-START
        def check_threat(obs: MonitorObservation, prox_side: float, prox_front: float, ttc_thresh: float, ttc_clear: float) -> bool:
            if len(obs.obs.r) == 0:
                return False
            clearance = obs.obs.r - obs.obs.sz
            v_approach = -obs.obs.v_r
            ttc = clearance / np.clip(v_approach, 0.1, 10.0)

            front = np.abs(obs.obs.az) < 2.0
            prox_danger = (clearance < prox_side) | (front & (clearance < prox_front)) | (front & (ttc < ttc_thresh) & (clearance < ttc_clear))

            return bool(np.any(prox_danger))

        def t_0_to_2(obs: MonitorObservation) -> bool:
            return check_threat(obs, 1.5, 2.1, 1.4, 5.0)

        def t_0_to_1(obs: MonitorObservation) -> bool:
            return check_threat(obs, 1.9, 2.6, 1.8, 6.0)

        def t_1_to_2(obs: MonitorObservation) -> bool:
            return check_threat(obs, 1.5, 2.1, 1.4, 5.0)

        def t_1_to_0(obs: MonitorObservation) -> bool:
            return not check_threat(obs, 2.1, 2.9, 2.0, 6.5)

        def t_2_to_0(obs: MonitorObservation) -> bool:
            return not check_threat(obs, 2.1, 2.9, 2.0, 6.5)

        def t_2_to_1(obs: MonitorObservation) -> bool:
            return not check_threat(obs, 1.7, 2.4, 1.6, 5.5)

        return [
            (0, 2, t_0_to_2),
            (0, 1, t_0_to_1),
            (1, 2, t_1_to_2),
            (1, 0, t_1_to_0),
            (2, 0, t_2_to_0),
            (2, 1, t_2_to_1)
        ]
        ## EVOLVE-BLOCK-END

    @property
    def initial_mode(self):
        ## EVOLVE-BLOCK-START
        def f(state: State) -> int:
            return 0

        return f
        ## EVOLVE-BLOCK-END
