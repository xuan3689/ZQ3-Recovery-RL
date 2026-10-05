"""Rigid-body / point-mass flight dynamics of the reusable first stage.

Model
-----
A 3-DOF translational model (position + velocity) coupled to a reduced
attitude model (tilt from the vertical + azimuth) that reproduces the two
effects that matter for recovery guidance:

* the thrust vector is fixed to the body axis, so a tilt error directly
  mis-directs the available control authority;
* the attitude loop is a rate-limited second-order autopilot, so the commanded
  direction is not reached instantaneously.

This is the "教学级简化模型" of the course manual, but the coupling between
attitude and translation is kept explicit because the whole point of the
residual-RL controller is to exploit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .config import G0, Config, MissionConfig, RocketConfig, WindConfig, air_density

DEG = np.pi / 180.0


# --------------------------------------------------------------------------- #
#  Frames
# --------------------------------------------------------------------------- #
def body_from_world(tilt: float, azim: float) -> np.ndarray:
    """Rotation matrix ``R_bw`` mapping world vectors into the body frame.

    The body frame is built from the thrust axis ``b = (sin t cos a, cos t,
    sin t sin a)``; the two transverse axes are chosen so that ``R_bw`` is a
    proper rotation (det = +1) and continuous in ``tilt`` around zero.
    """
    st, ct = np.sin(tilt), np.cos(tilt)
    ca, sa = np.cos(azim), np.sin(azim)
    b = np.array([st * ca, ct, st * sa])  # body x (thrust axis)
    # a reference "up" that is never parallel to b
    ref = np.array([0.0, 1.0, 0.0]) if abs(ct) < 0.999 else np.array([1.0, 0.0, 0.0])
    c = np.cross(ref, b)
    c /= max(np.linalg.norm(c), 1e-9)  # transverse axis
    d = np.cross(b, c)  # remaining axis
    # rows [b, c, b x c] give det = b . (c x (b x c)) = b . b = +1
    return np.vstack([b, c, d])


# --------------------------------------------------------------------------- #
#  Wind field
# --------------------------------------------------------------------------- #
class WindField:
    """Randomised wind: mean profile, a low-altitude shear layer and an
    Ornstein-Uhlenbeck turbulence process.

    The field is a deterministic function of time and altitude once the
    parameters are drawn, which lets the estimator identify it online and the
    predictor extrapolate it.
    """

    def __init__(self, cfg: WindConfig, rng: np.random.Generator):
        self.cfg = cfg
        if not cfg.enable:
            self.speed = 0.0
            self.direction = 0.0
            self.turb = np.zeros(3)
            return
        lo, hi = cfg.base_speed_range
        self.speed = float(rng.uniform(lo, hi))
        dlo, dhi = cfg.direction_range
        self.direction = float(rng.uniform(dlo, dhi))
        # OU turbulence state (3 components), started at stationarity
        self.turb = rng.normal(0.0, cfg.turb_sigma, size=3)
        self._last_t = 0.0

    def _profile(self, altitude: np.ndarray | float) -> np.ndarray:
        """Mean horizontal wind magnitude as a function of altitude."""
        h = np.asarray(altitude, dtype=float)
        shear = 1.0 + self.cfg.shear_gain * np.exp(
            -0.5 * ((h - self.cfg.shear_alt) / 90.0) ** 2
        )
        # weak power-law boundary-layer growth below the shear layer
        bl = 1.0 + 0.25 * np.exp(-np.maximum(h, 0.0) / 260.0)
        return self.speed * shear * bl

    def advance(self, t: float, dt: float, rng: np.random.Generator) -> None:
        """Integrate the OU turbulence one step (exact discretisation)."""
        if not self.cfg.enable:
            return
        if dt <= 0.0:
            return
        theta = np.exp(-dt / self.cfg.turb_tau)
        sigma = self.cfg.turb_sigma * np.sqrt(max(1.0 - theta * theta, 0.0))
        self.turb = theta * self.turb + rng.normal(0.0, sigma, size=3)
        self._last_t = t

    def velocity(self, altitude: np.ndarray | float) -> np.ndarray:
        """Wind velocity vector [m/s] at a given altitude (3,)."""
        if not self.cfg.enable:
            return np.zeros(3)
        mag = float(np.asarray(self._profile(altitude)))
        cy, sy = np.cos(self.direction), np.sin(self.direction)
        mean = np.array([mag * cy, 0.0, mag * sy])
        return mean + self.turb


# --------------------------------------------------------------------------- #
#  Flight state
# --------------------------------------------------------------------------- #
@dataclass
class FlightState:
    """Full (ground-truth) vehicle state used by the simulator."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 30000.0
    # attitude: tilt from the +y (up) axis and azimuth of the tilt direction
    tilt: float = 0.0
    tilt_rate: float = 0.0
    azim: float = 0.0
    # engine
    throttle: float = 0.0
    engine_on: bool = False

    def body_axis(self) -> np.ndarray:
        """Unit vector of the body axis (thrust direction) in world frame."""
        st, ct = np.sin(self.tilt), np.cos(self.tilt)
        return np.array([st * np.cos(self.azim), ct, st * np.sin(self.azim)])

    def copy(self) -> "FlightState":
        return FlightState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            mass=self.mass,
            tilt=self.tilt,
            tilt_rate=self.tilt_rate,
            azim=self.azim,
            throttle=self.throttle,
            engine_on=self.engine_on,
        )


# --------------------------------------------------------------------------- #
#  Actuator command
# --------------------------------------------------------------------------- #
@dataclass
class ActuatorCommand:
    """Low-level command produced by the guidance / safety shield."""

    throttle: float = 0.0
    tilt: float = 0.0  # commanded tilt from vertical [rad]
    azim: float = 0.0  # commanded azimuth [rad]
    engine_on: bool = True


# --------------------------------------------------------------------------- #
#  Plant
# --------------------------------------------------------------------------- #
class VehicleDynamics:
    """Non-linear plant with 100 Hz-capable integration.

    ``thrust_scale`` is the per-episode delivered-to-rated thrust ratio.  It is
    applied to the *actual* thrust produced by the engine, while
    :meth:`available_accel` deliberately keeps reporting the **rated** envelope:
    the guidance law plans against the rated vehicle and therefore does not know
    the scale.  The discrepancy shows up in the accelerometer and is absorbed by
    the navigation EKF's disturbance state, which is exactly how a real vehicle
    would detect a weak engine.
    """

    def __init__(self, rocket: RocketConfig, mission: MissionConfig, cfg: Config,
                 thrust_scale: float = 1.0):
        self.rocket = rocket
        self.mission = mission
        self.cfg = cfg
        self.thrust_scale = float(thrust_scale)

    # ------------------------------------------------------------------ #
    def available_accel(self, state: FlightState) -> Tuple[float, float]:
        """(min, max) *rated* thrust acceleration at the current mass.

        Reports the rated envelope, not the delivered one -- see the class
        docstring.  This is what the guidance law is allowed to believe.
        """
        a_max = self.rocket.thrust_max / state.mass
        a_min = self.rocket.throttle_min * a_max
        return a_min, a_max

    # ------------------------------------------------------------------ #
    def delivered_accel(self, state: FlightState) -> float:
        """Actual full-throttle thrust acceleration including degradation."""
        return self.rocket.thrust_max * self.thrust_scale / state.mass

    # ------------------------------------------------------------------ #
    def derivatives(
        self,
        state: FlightState,
        cmd: ActuatorCommand,
        wind_vel: np.ndarray,
        throttle_cmd: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Continuous-time derivative of (pos, vel, mass, tilt, tilt_rate).

        The attitude autopilot is a second-order tracker with a rate limit, and
        the engine is a first-order lag; both are what the RL residual must
        learn to work around.
        """
        rk = self.rocket
        rho = float(air_density(state.pos[1]))
        rel = state.vel - wind_vel

        # --- aero --------------------------------------------------------
        v_mag = float(np.linalg.norm(rel))
        drag_acc = -0.5 * rho * v_mag * rel * rk.cd_a / state.mass

        # --- thrust ------------------------------------------------------
        # the *delivered* thrust is what actually accelerates the vehicle
        thrust = throttle_cmd * rk.thrust_max * self.thrust_scale
        a_thrust = thrust / state.mass
        accel = a_thrust * state.body_axis() + drag_acc
        accel[1] -= G0

        # --- mass --------------------------------------------------------
        mdot = -thrust / rk.exhaust_velocity

        # --- attitude autopilot -----------------------------------------
        # slew limit on the commanded tilt is applied here (not in the shield)
        err = cmd.tilt - state.tilt
        tilt_acc = rk.att_wn ** 2 * err - 2.0 * rk.att_zeta * rk.att_wn * state.tilt_rate
        rate = state.tilt_rate
        rate_max = rk.tilt_rate_max_deg * DEG
        if abs(rate) > rate_max:
            rate = np.sign(rate) * rate_max
            tilt_acc = min(tilt_acc, 0.0) if state.tilt_rate > 0 else max(tilt_acc, 0.0)
        return state.vel, accel, mdot, rate, tilt_acc

    # ------------------------------------------------------------------ #
    def step(
        self,
        state: FlightState,
        cmd: ActuatorCommand,
        wind_vel: np.ndarray,
        dt: float,
        engine_lag: float,
    ) -> FlightState:
        """Advance the plant by ``dt`` with RK4 on the translational part and
        semi-implicit Euler on the (cheap) rotational part."""
        # first-order engine lag on throttle
        target = cmd.throttle if cmd.engine_on else 0.0
        alpha = min(dt / max(engine_lag, 1e-6), 1.0)
        throttle = state.throttle + alpha * (target - state.throttle)
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # RK4 over the translational subsystem, frozen attitude
        def f(s: FlightState, thr: float):
            _, acc, mdot, _, _ = self.derivatives(s, cmd, wind_vel, thr)
            return acc, mdot

        s0 = state.copy()
        s0.throttle = throttle
        a1, m1 = f(s0, throttle)
        s2 = state.copy(); s2.pos = state.pos + 0.5 * dt * state.vel
        s2.vel = state.vel + 0.5 * dt * a1; s2.mass = state.mass + 0.5 * dt * m1
        a2, m2 = f(s2, throttle)
        s3 = state.copy(); s3.pos = state.pos + 0.5 * dt * state.vel
        s3.vel = state.vel + 0.5 * dt * a2; s3.mass = state.mass + 0.5 * dt * m2
        a3, m3 = f(s3, throttle)
        s4 = state.copy(); s4.pos = state.pos + dt * state.vel
        s4.vel = state.vel + dt * a3; s4.mass = state.mass + dt * m3
        a4, m4 = f(s4, throttle)

        new = state.copy()
        new.pos = state.pos + dt / 6.0 * (state.vel + 2 * s2.vel + 2 * s3.vel + s4.vel)
        new.vel = state.vel + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        new.mass = float(max(state.mass + dt / 6.0 * (m1 + 2 * m2 + 2 * m3 + m4),
                             self.rocket.m_dry))

        # rotational update
        _, _, _, rate, tilt_acc = self.derivatives(state, cmd, wind_vel, throttle)
        new.tilt_rate = float(np.clip(state.tilt_rate + tilt_acc * dt,
                                      -self.rocket.tilt_rate_max_deg * DEG,
                                      self.rocket.tilt_rate_max_deg * DEG))
        new.tilt = float(state.tilt + new.tilt_rate * dt)
        new.azim = float(cmd.azim)
        new.throttle = throttle
        new.engine_on = bool(cmd.engine_on and throttle > 1e-6)
        return new


# --------------------------------------------------------------------------- #
#  Termination helpers
# --------------------------------------------------------------------------- #
def touchdown_metrics(state: FlightState, mission: MissionConfig) -> dict:
    """Terminal metrics used for both reward shaping and evaluation."""
    dx = state.pos[0] - mission.target[0]
    dz = state.pos[2] - mission.target[2]
    lateral = float(np.hypot(dx, dz))
    speed = float(np.linalg.norm(state.vel))
    tilt_deg = float(abs(state.tilt) / DEG)
    ok = (
        lateral <= mission.pos_tol
        and speed <= mission.vel_tol
        and tilt_deg <= mission.tilt_tol_deg
    )
    return {
        "lateral": lateral,
        "speed": speed,
        "tilt_deg": tilt_deg,
        "v_y": float(state.vel[1]),
        "success": bool(ok),
    }
