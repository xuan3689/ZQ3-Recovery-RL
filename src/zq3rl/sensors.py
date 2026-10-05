"""Multi-source sensor models: IMU, GNSS and barometric altimeter.

Each sensor is sampled at its own rate and carries the error sources that the
course manual lists for the real vehicle: white noise, turn-on bias, random
walk, and (for GNSS) a slow update rate.  The estimators in ``estimator.py``
consume these measurements; nothing here is filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import G0, SensorConfig, air_density
from .dynamics import FlightState, WindField, body_from_world

DEG = np.pi / 180.0


@dataclass
class ImuSample:
    accel: np.ndarray  # specific force in the *body* frame [m/s^2]
    gyro: np.ndarray  # angular rate in the body frame [rad/s]
    t: float


@dataclass
class GnssSample:
    pos: np.ndarray  # world-frame position [m]
    vel: np.ndarray  # world-frame velocity [m/s]
    t: float


@dataclass
class BaroSample:
    altitude: float
    t: float


class SensorSuite:
    """Stateful sensor simulator (biases and random walks persist in time)."""

    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        # turn-on biases
        self.accel_bias = rng.normal(0.0, cfg.accel_bias0, size=3)
        self.gyro_bias = rng.normal(0.0, cfg.gyro_bias0_deg * DEG, size=3)
        # scheduling
        self._next_imu = 0.0
        self._next_gnss = 0.0
        self._next_baro = 0.0
        # last published
        self.imu = ImuSample(np.zeros(3), np.zeros(3), 0.0)
        self.gnss = GnssSample(np.zeros(3), np.zeros(3), 0.0)
        self.baro = BaroSample(0.0, 0.0)
        self.imu_count = 0
        self.gnss_count = 0
        self.baro_count = 0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        cfg, rng = self.cfg, self.rng
        self.accel_bias = rng.normal(0.0, cfg.accel_bias0, size=3)
        self.gyro_bias = rng.normal(0.0, cfg.gyro_bias0_deg * DEG, size=3)
        self._next_imu = self._next_gnss = self._next_baro = 0.0
        self.imu_count = self.gnss_count = self.baro_count = 0

    # ------------------------------------------------------------------ #
    def _accel_available(self, state: FlightState, wind_vel: np.ndarray,
                         throttle: float, rocket, thrust_scale: float = 1.0) -> np.ndarray:
        """True specific force (what an ideal IMU would read), body frame.

        The specific force is ``(thrust + drag) / m`` resolved in the body
        frame.  Its *transverse* components are non-zero only because the drag
        vector is not aligned with the body axis, which is precisely what makes
        the attitude observable from an accelerometer once the relative-wind
        direction is known from GNSS.

        ``thrust_scale`` is the delivered-to-rated ratio: the accelerometer sees
        the *delivered* thrust, so a degraded engine is observable here even
        though the guidance law assumes the rated value.
        """
        rho = float(air_density(state.pos[1]))
        rel = state.vel - wind_vel
        v_mag = float(np.linalg.norm(rel))
        drag_acc = -0.5 * rho * v_mag * rel * rocket.cd_a / state.mass
        thrust_acc = throttle * rocket.thrust_max * thrust_scale / state.mass
        nongrav = thrust_acc * state.body_axis() + drag_acc
        return body_from_world(state.tilt, state.azim) @ nongrav

    # ------------------------------------------------------------------ #
    def step(self, t: float, state: FlightState, wind_vel: np.ndarray,
             throttle: float, rocket, thrust_scale: float = 1.0) -> Dict[str, object]:
        """Advance the sensors to time ``t`` and return the fresh samples."""
        cfg = self.cfg
        if not cfg.enable:
            # perfect sensing
            self.imu = ImuSample(
                np.array([throttle * rocket.thrust_max / state.mass, 0.0, 0.0]),
                np.zeros(3), t)
            self.gnss = GnssSample(state.pos.copy(), state.vel.copy(), t)
            self.baro = BaroSample(float(state.pos[1]), t)
            self.imu_count += 1
            self.gnss_count += 1
            self.baro_count += 1
            return {"imu": True, "gnss": True, "baro": True}

        fresh = {"imu": False, "gnss": False, "baro": False}
        dt = cfg  # alias for readability below

        if t >= self._next_imu - 1e-9:
            period = 1.0 / cfg.imu_rate
            n = max(int(round((t - self._next_imu) / period)) + 1, 1)
            self._next_imu += n * period
            # random walks
            self.accel_bias += self.rng.normal(0.0, cfg.accel_bias_rw,
                                               size=3) * np.sqrt(n * period)
            self.gyro_bias += self.rng.normal(0.0, cfg.gyro_bias_rw_deg * DEG,
                                              size=3) * np.sqrt(n * period)
            acc = self._accel_available(state, wind_vel, throttle, rocket, thrust_scale)
            acc = acc + self.accel_bias + self.rng.normal(0.0, cfg.accel_noise, 3)
            gyr = np.array([state.tilt_rate, 0.0, 0.0]) + self.gyro_bias + \
                self.rng.normal(0.0, cfg.gyro_noise_deg * DEG, 3)
            self.imu = ImuSample(acc, gyr, t)
            self.imu_count += 1
            fresh["imu"] = True

        if t >= self._next_gnss - 1e-9:
            period = 1.0 / cfg.gnss_rate
            n = max(int(round((t - self._next_gnss) / period)) + 1, 1)
            self._next_gnss += n * period
            self.gnss = GnssSample(
                state.pos + self.rng.normal(0.0, cfg.gnss_pos_noise, 3),
                state.vel + self.rng.normal(0.0, cfg.gnss_vel_noise, 3),
                t)
            self.gnss_count += 1
            fresh["gnss"] = True

        if t >= self._next_baro - 1e-9:
            period = 1.0 / cfg.baro_rate
            n = max(int(round((t - self._next_baro) / period)) + 1, 1)
            self._next_baro += n * period
            self.baro = BaroSample(
                float(state.pos[1] + self.rng.normal(0.0, cfg.baro_noise)), t)
            self.baro_count += 1
            fresh["baro"] = True

        return fresh
