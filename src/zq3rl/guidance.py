"""Guidance: a layered baseline plus predictive-perception compensation and a
bounded reinforcement-learning residual.

Why layered
-----------
A single polynomial (Apollo "E-guidance") law pins *one* time-to-go for all
three axes.  For a vertical-recovery entry that is over-constrained: the
vertical channel needs ``T ≈ 19 s`` to bleed a 150 m/s sink rate over 1.5 km,
while the lateral channel needs ``T ≈ 31 s`` to null a 400 m downrange offset,
and no single ``T`` satisfies both.  The course manual resolves this the way
real vehicles do -- with a *layered* architecture (``4.3.4 控制体系耦合特性
与控制原则``):

* **vertical channel** tracks an explicit descent-speed profile, which is
  exactly the manual's ``a_T = a_ref + k_v (v_ref - v)`` speed-feedback law;
* **lateral channel** is a cascade (position -> commanded velocity -> commanded
  acceleration), the manual's ``T_cmd = K_px Δx + K_dx Δv + mg`` double loop;
* **attitude** is slaved to the direction of the resulting thrust vector, and
  the safety shield keeps it inside the altitude-scheduled authority.

The invention is the two terms added on top of this baseline:

1. ``a_dist_ff`` -- a **predictive-perception** feed-forward built from the
   disturbance *forecast* (not just its instantaneous estimate);
2. ``a_rl`` -- a **bounded residual** from a PPO policy trained with a
   curriculum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import G0, Config, GuidanceConfig
from .dynamics import ActuatorCommand

DEG = np.pi / 180.0


# --------------------------------------------------------------------------- #
#  Baseline: layered guidance
# --------------------------------------------------------------------------- #
@dataclass
class GuidanceSolution:
    accel: np.ndarray  # commanded *net* thrust acceleration, world frame [m/s^2]
    tgo: float  # equivalent time-to-go, used as an RL observation [s]
    feasible: bool  # whether the command fits the thrust envelope
    v_ref: float = 0.0  # reference sink rate used by the vertical channel


def descent_speed_profile(h: float, v_target: float, a_ref: float,
                          v_max: float) -> Tuple[float, float]:
    """Reference sink rate and its altitude derivative.

    ``v_ref(h) = -min( v_max, sqrt( v_target^2 + 2 a_ref h ) )``

    Starting from the terminal sink rate ``v_target`` and working upward, the
    profile is the fastest descent that can still be arrested at ``a_ref``.  It
    is monotonically decreasing toward the target, so a speed-feedback loop
    tracking it can never be driven into the positive-feedback runaway that a
    "solve for the exact final state" law suffers from.

    Returns ``(v_ref, dv_ref/dh)``.  The derivative is zero inside the
    saturated region and ``-a_ref / |v_ref|`` outside it; the guidance law
    needs it to build a consistent feed-forward.
    """
    h = max(h, 0.0)
    s = math.sqrt(max(v_target * v_target + 2.0 * a_ref * h, 0.0))
    if s >= v_max:
        return -v_max, 0.0
    return -s, (-a_ref / max(s, 1e-6))


def lateral_cascade(
    dp: np.ndarray, vel_lat: np.ndarray, kp: float, kd: float, a_lat_max: float
) -> np.ndarray:
    """Cascade lateral controller: position -> velocity -> acceleration.

    ``dp`` is the position error ``target - pos`` and ``vel_lat`` the *actual*
    horizontal velocity.  The outer loop turns position error into a commanded
    horizontal velocity (saturated so the vehicle never builds an
    un-arrestable lateral rate); the inner loop turns the velocity error
    ``v_cmd - vel_lat`` into a horizontal acceleration.
    """
    v_cmd = kp * dp
    v_mag = float(np.linalg.norm(v_cmd))
    if v_mag > 30.0:
        v_cmd = v_cmd / v_mag * 30.0
    a = kd * (v_cmd - vel_lat)
    a_mag = float(np.linalg.norm(a))
    if a_mag > a_lat_max:
        a = a / a_mag * a_lat_max
    return a


def baseline_guidance(
    pos: np.ndarray,
    vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    accel_min: float,
    accel_max: float,
    cfg: GuidanceConfig,
    gravity: float = G0,
    tilt_authority: float = 25.0 * DEG,
) -> GuidanceSolution:
    """Layered powered-descent baseline (net acceleration, world frame).

    The vertical channel tracks a descent-speed profile with an analytic
    feed-forward and a proportional term, so it adapts to mass depletion and to
    the vertical component of the disturbance.  The horizontal channel is a
    position -> velocity -> acceleration cascade.  The two channels share one
    thrust vector, so the vertical channel must reserve authority for the
    lateral one.
    """
    dp = target_pos - pos
    dv = target_vel - vel
    h = float(pos[1] - target_pos[1])

    cos_t = math.cos(tilt_authority)
    sin_t = math.sin(tilt_authority)
    a_y_cap = max(accel_max * cos_t - gravity, 0.5)  # reserve for lateral
    a_lat_cap = max(accel_max * sin_t, 0.5)

    # ---- vertical channel ------------------------------------------------
    # Tracking ``v_ref(h)`` exactly requires the net acceleration
    #     dv/dt = (dv_ref/dh) * v
    # which is the feed-forward term; a proportional term on the velocity
    # error removes the residual.  The profile deceleration is chosen with
    # margin below the vertical budget so the loop can always track it.
    a_ref = 0.55 * a_y_cap
    v_ref, dv_ref_dh = descent_speed_profile(h, float(target_vel[1]),
                                             a_ref=a_ref, v_max=60.0)
    a_ff = float(vel[1]) * dv_ref_dh
    a_y = a_ff + 2.6 * (v_ref - float(vel[1]))
    a_y = float(np.clip(a_y, -(gravity - 1.0), a_y_cap))

    # ---- lateral: cascade -------------------------------------------------
    a_lat = lateral_cascade(dp[0::2], vel[0::2], kp=0.32, kd=1.10,
                            a_lat_max=a_lat_cap)
    # an upward-thrusting vehicle can only push sideways in proportion to its
    # vertical thrust, so bound the lateral command by the cone
    cone = max(a_y + gravity, 0.0) * math.tan(tilt_authority)
    if float(np.linalg.norm(a_lat)) > cone > 1e-9:
        a_lat = a_lat / float(np.linalg.norm(a_lat)) * cone

    accel = np.array([a_lat[0], a_y, a_lat[1]])

    tgo = _equivalent_tgo(h, float(vel[1]), float(target_vel[1]), a_y)
    feasible = _is_feasible(accel, accel_min, accel_max, gravity)
    return GuidanceSolution(accel=accel, tgo=tgo, feasible=feasible, v_ref=v_ref)


def _equivalent_tgo(h: float, vy: float, v_target: float, a_y: float) -> float:
    """Time to reach the pad under the current vertical command (for the RL
    observation and for logging)."""
    a = a_y  # net vertical acceleration; gravity already inside the command
    disc = vy * vy + 2.0 * a * h
    if disc < 0.0:
        return 60.0
    r = math.sqrt(disc)
    cands = [(-vy + r) / a if abs(a) > 1e-6 else 60.0,
             (-vy - r) / a if abs(a) > 1e-6 else 60.0]
    pos_c = [c for c in cands if c > 0.05]
    return float(min(pos_c)) if pos_c else 60.0


def _is_feasible(accel: np.ndarray, accel_min: float, accel_max: float,
                 gravity: float) -> bool:
    tv = accel.copy(); tv[1] += gravity
    return accel_min * 0.98 <= float(np.linalg.norm(tv)) <= accel_max * 1.001


# --------------------------------------------------------------------------- #
#  Safety shield
# --------------------------------------------------------------------------- #
@dataclass
class ShieldReport:
    clipped_throttle: bool = False
    clipped_tilt: bool = False
    clipped_residual: bool = False
    reason: str = ""


class SafetyShield:
    """Projects the total command onto the physically admissible set.

    Guarantees, in order:
    1. the residual action stays inside its box;
    2. the thrust vector has a strictly positive vertical component (the engine
       cannot pull the vehicle down);
    3. the thrust magnitude stays inside ``[accel_min, accel_max]``;
    4. the tilt stays inside the altitude-scheduled attitude authority.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def tilt_limit(self, altitude: float) -> float:
        rk = self.cfg.rocket
        a = rk.tilt_max_deg * DEG
        b = rk.tilt_max_terminal_deg * DEG
        h = float(np.clip((altitude - 60.0) / 240.0, 0.0, 1.0))
        return b + (a - b) * h

    def project(
        self,
        accel_total: np.ndarray,
        accel_min: float,
        accel_max: float,
        altitude: float,
        residual_norm_before: float,
    ) -> Tuple[np.ndarray, ActuatorCommand, ShieldReport]:
        rep = ShieldReport()
        g = self.cfg.guidance

        if residual_norm_before > g.residual_max * 1.001:
            rep.clipped_residual = True

        thrust_vec = accel_total.copy()
        thrust_vec[1] += G0

        # 2. positive vertical thrust component
        min_vert = 0.30 * accel_min
        if thrust_vec[1] < min_vert:
            thrust_vec[1] = min_vert
            rep.clipped_tilt = True
            rep.reason = "vertical thrust floor"

        # 3. thrust magnitude
        mag = float(np.linalg.norm(thrust_vec))
        if mag < 1e-9:
            mag = 1e-9
        if mag < accel_min:
            thrust_vec = thrust_vec / mag * accel_min
            rep.clipped_throttle = True
            rep.reason = "thrust below minimum"
        elif mag > accel_max:
            thrust_vec = thrust_vec / mag * accel_max
            rep.clipped_throttle = True
            rep.reason = "thrust saturated"

        # 4. tilt authority
        limit = self.tilt_limit(altitude)
        vmag = max(float(np.linalg.norm(thrust_vec)), 1e-9)
        tilt = math.acos(float(np.clip(thrust_vec[1] / vmag, -1.0, 1.0)))
        if tilt > limit:
            horiz = thrust_vec.copy(); horiz[1] = 0.0
            hn = float(np.linalg.norm(horiz))
            if hn > 1e-9:
                target_h = math.tan(limit) * abs(thrust_vec[1])
                horiz = horiz / hn * target_h
            thrust_vec = np.array([horiz[0], thrust_vec[1], horiz[2]])
            rep.clipped_tilt = True
            rep.reason = "attitude authority"

        net = thrust_vec.copy()
        net[1] -= G0
        throttle = float(np.clip(np.linalg.norm(thrust_vec) / accel_max, 0.0, 1.0))
        horiz = math.hypot(thrust_vec[0], thrust_vec[2])
        azim = math.atan2(thrust_vec[2], thrust_vec[0]) if horiz > 1e-6 else 0.0
        tilt_cmd = math.atan2(horiz, max(thrust_vec[1], 1e-9))
        cmd = ActuatorCommand(throttle=throttle, tilt=tilt_cmd, azim=azim, engine_on=True)
        return net, cmd, rep


# --------------------------------------------------------------------------- #
#  Full controller
# --------------------------------------------------------------------------- #
class GuidanceController:
    """Baseline + predictive feed-forward + bounded RL residual + shield.

    The computation is split into two stages so that the disturbance *forecast*
    can be built from the baseline command.  This ordering matters: a forward
    model integrated with zero commanded acceleration would let the vehicle
    appear to accelerate at ``g`` for the whole horizon, which inflates the
    predicted relative velocity and therefore the predicted drag.  Feeding the
    baseline command into the predictor keeps the forecast consistent with the
    trajectory the vehicle is actually going to fly.
    """

    def __init__(
        self,
        cfg: Config,
        use_prediction: bool = True,
    ):
        self.cfg = cfg
        self.use_prediction = use_prediction
        self.shield = SafetyShield(cfg)
        self.last_solution: Optional[GuidanceSolution] = None

    # ------------------------------------------------------------------ #
    def compute_baseline(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        accel_min: float,
        accel_max: float,
    ) -> GuidanceSolution:
        """Stage 1: the analytic layered guidance command."""
        cfg = self.cfg
        sol = baseline_guidance(
            pos, vel,
            np.asarray(cfg.mission.target, dtype=float),
            np.array([0.0, cfg.mission.v_target_y, 0.0]),
            accel_min, accel_max, cfg.guidance,
        )
        self.last_solution = sol
        return sol

    # ------------------------------------------------------------------ #
    def finalize(
        self,
        sol: GuidanceSolution,
        disturbance_pred: np.ndarray,
        residual: np.ndarray,
        accel_min: float,
        accel_max: float,
        pos: np.ndarray,
    ) -> Tuple[np.ndarray, ActuatorCommand, ShieldReport]:
        """Stage 2: add the feed-forward and residual, then project."""
        g = self.cfg.guidance
        accel = sol.accel.copy()

        # ---- predictive-perception feed-forward --------------------------
        # Gain-scheduled in altitude.  High up, the vehicle has tens of seconds
        # of flight left, so the feedback cascade already nulls the lateral
        # error and an extra feed-forward only fights it.  Near the ground the
        # remaining time is shorter than the disturbance time constant, so
        # feedback can no longer catch up and the *forecast* is the only way to
        # counter a gust before it acts.  The schedule therefore fades the term
        # in below the altitude where reaction time runs out.
        if self.use_prediction and g.use_disturbance_ff:
            h = float(pos[1] - self.cfg.mission.target[1])
            sched = float(np.clip((g.ff_schedule_alt - h)
                                  / max(g.ff_schedule_alt - g.ff_full_alt, 1e-6),
                                  0.0, 1.0))
            if sched > 0.0:
                ff = -g.dist_ff_gain * sched * np.asarray(disturbance_pred, dtype=float)
                ff = np.clip(ff, -g.residual_max, g.residual_max)
                accel = accel + ff

        # ---- bounded RL residual ----------------------------------------
        r = np.asarray(residual, dtype=float)
        r_norm = float(np.linalg.norm(r))
        if r_norm > g.residual_max:
            r = r / r_norm * g.residual_max
        accel = accel + r

        return self.shield.project(accel, accel_min, accel_max, pos[1], r_norm)

    # ------------------------------------------------------------------ #
    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: float,
        disturbance: np.ndarray,
        disturbance_pred: np.ndarray,
        residual: np.ndarray,
        accel_min: float,
        accel_max: float,
    ) -> Tuple[np.ndarray, ActuatorCommand, ShieldReport]:
        """Convenience wrapper running both stages with an external forecast."""
        sol = self.compute_baseline(pos, vel, accel_min, accel_max)
        return self.finalize(sol, disturbance_pred, residual,
                             accel_min, accel_max, pos)
