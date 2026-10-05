"""Predictive perception: multi-source state estimation and disturbance
prediction.

Three estimators run in parallel and their outputs are what the guidance law
and the RL residual policy actually see:

``NavigationEKF``
    9-state EKF over ``[p, v, a_b]`` where ``a_b`` is a lumped un-modelled
    acceleration (aero + wind + thrust mismatch).  Fuses 100 Hz IMU
    propagation with 10 Hz GNSS and 20 Hz barometric updates.  This is the
    "多源融合滤波" of the manual, made explicit.

``AttitudeEKF``
    4-state EKF over ``[tilt, azim, gyro_bias_x, gyro_bias_z]``.  It fuses the
    gyro, the specific-force direction (attitude observability through drag)
    and the known tilt command.

``DisturbancePredictor``
    Turns the EKF's disturbance estimate into a *forward prediction* of the
    wind-induced acceleration over the remaining flight.  This is the
    "预测感知" contribution: the RL policy is given a forecast of the
    disturbance it must reject, not just the instantaneous error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .config import G0, Config, EstimatorConfig
from .dynamics import FlightState, WindField, body_from_world, air_density

DEG = np.pi / 180.0


# --------------------------------------------------------------------------- #
#  Small linear-algebra helpers
# --------------------------------------------------------------------------- #
def _sym(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)


def _safe_inv(S: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_sym(S) + 1e-9 * np.eye(S.shape[0]))


# --------------------------------------------------------------------------- #
#  Navigation EKF
# --------------------------------------------------------------------------- #
class NavigationEKF:
    """Constant-acceleration + lumped-disturbance navigation filter.

    State ``x = [p(3), v(3), a_b(3)]`` in the world frame.  ``a_b`` absorbs
    aerodynamic drag, wind and any thrust-model mismatch; because the manual's
    plant applies drag *continuously*, ``a_b`` is modelled as a random walk so
    the filter can track it rather than assuming it is constant.
    """

    def __init__(self, cfg: EstimatorConfig, dt: float = 0.01):
        self.cfg = cfg
        self.dt = dt
        self.x = np.zeros(9)
        self.P = np.diag(
            [cfg.pos_init_std ** 2] * 3
            + [cfg.vel_init_std ** 2] * 3
            + [cfg.bias_init_std ** 2] * 3
        )
        self._F = self._build_F(dt)
        self._Q = self._build_Q(dt)
        self.last_gnss_t = -1e9
        self.last_baro_t = -1e9
        self.nis_gnss: list[float] = []
        self.nis_baro: list[float] = []

    # ------------------------------------------------------------------ #
    def _build_F(self, dt: float) -> np.ndarray:
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt * dt
        F[3:6, 6:9] = np.eye(3) * dt
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        qa = self.cfg.accel_noise_std ** 2
        qb = self.cfg.bias_rw_std ** 2
        Q = np.zeros((9, 9))
        # white acceleration on velocity, projected into position
        Qv = qa * np.eye(3) * dt
        Qp = qa * np.eye(3) * (dt ** 3 / 3.0)
        Qpv = qa * np.eye(3) * (0.5 * dt * dt)
        Q[0:3, 0:3] = Qp
        Q[3:6, 3:6] = Qv
        Q[0:3, 3:6] = Qpv
        Q[3:6, 0:3] = Qpv
        Q[6:9, 6:9] = qb * np.eye(3) * dt
        return Q

    # ------------------------------------------------------------------ #
    def predict(self, accel_cmd: np.ndarray, dt: float) -> None:
        """Propagate with the *known* commanded acceleration.

        ``accel_cmd`` is the full known world-frame acceleration, i.e. the
        commanded thrust acceleration **plus gravity**.  The unknown part is
        carried by the ``a_b`` state, so the predicted total acceleration is
        ``accel_cmd + a_b``.
        """
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        a_total = accel_cmd + self.x[6:9]
        self.x[0:3] += self.x[3:6] * dt + 0.5 * a_total * dt * dt
        self.x[3:6] += a_total * dt
        self.P = _sym(F @ self.P @ F.T + Q)

    # ------------------------------------------------------------------ #
    def update_accel(self, specific_force: np.ndarray,
                     accel_cmd_thrust: np.ndarray) -> None:
        """Pseudo-measurement of the disturbance state from the accelerometer.

        The IMU reads the specific force ``s = a_thrust + a_drag`` (gravity is
        excluded by definition).  Subtracting the *commanded* thrust
        acceleration leaves a direct observation of the disturbance state
        ``a_b = a_drag``.  This is what makes the wind identifiable online.

        The measurement noise is the accelerometer noise *plus* the uncertainty
        in the thrust actually delivered (engine lag and thrust-model error),
        which is the dominant term -- using only the IMU noise would make the
        filter trust the accelerometer far too much and bias the velocity
        estimate through the correlated command error.
        """
        z = specific_force - accel_cmd_thrust
        H = np.zeros((3, 9))
        H[0:3, 6:9] = np.eye(3)
        R = np.eye(3) * (self.cfg.accel_noise_std ** 2 + 0.30 ** 2)
        self.update(z, H, R)

    # ------------------------------------------------------------------ #
    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray,
               nis_log: Optional[list] = None) -> None:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ _safe_inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(9) - K @ H
        self.P = _sym(I_KH @ self.P @ I_KH.T + K @ R @ K.T)  # Joseph form
        if nis_log is not None:
            nis_log.append(float(y @ _safe_inv(S) @ y))

    def update_gnss(self, pos: np.ndarray, vel: np.ndarray, t: float) -> None:
        H = np.zeros((6, 9))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        R = np.diag([self.cfg.gnss_pos_std ** 2] * 3
                    + [self.cfg.gnss_vel_std ** 2] * 3)
        self.update(np.concatenate([pos, vel]), H, R, self.nis_gnss)
        self.last_gnss_t = t

    def update_baro(self, altitude: float, t: float) -> None:
        H = np.zeros((1, 9))
        H[0, 1] = 1.0
        R = np.array([[self.cfg.baro_std ** 2]])
        self.update(np.array([altitude]), H, R, self.nis_baro)
        self.last_baro_t = t

    # ------------------------------------------------------------------ #
    @property
    def pos(self) -> np.ndarray:
        return self.x[0:3]

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def disturbance(self) -> np.ndarray:
        """Estimated lumped acceleration not explained by the command [m/s^2]."""
        return self.x[6:9]

    def pos_std(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.P)[0:3], 0.0, None))

    def vel_std(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.P)[3:6], 0.0, None))


# --------------------------------------------------------------------------- #
#  Attitude EKF
# --------------------------------------------------------------------------- #
class AttitudeEKF:
    """Attitude filter over ``[tilt, tilt_rate, azim, b_gyro]``.

    The gyro alone cannot observe its own bias, and the accelerometer's
    transverse signal is far too weak (the thrust acceleration dwarfs the aero
    term) to serve as a tilt reference on its own.  What *is* known is the
    commanded tilt: the autopilot is a second-order tracker with a known
    natural frequency and damping, so the command is a legitimate model input.
    Propagating the tilt through that model and treating the gyro as a
    measurement of ``tilt_rate + b_gyro`` makes the bias observable, which is
    the standard complementary-filter structure.
    """

    def __init__(self, cfg: EstimatorConfig, dt: float = 0.01,
                 wn: float = 3.5, zeta: float = 0.9,
                 gyro_noise_deg: float = 0.05):
        self.cfg = cfg
        self.dt = dt
        self.wn = wn
        self.zeta = zeta
        self.gyro_noise_deg = gyro_noise_deg
        self.x = np.zeros(4)  # [tilt, tilt_rate, azim, b_gyro]
        self.P = np.diag([
            (cfg.att_init_std_deg * DEG) ** 2,
            (3.0 * DEG) ** 2,
            (cfg.att_init_std_deg * DEG) ** 2,
            (cfg.gyro_bias_init_std_deg * DEG) ** 2,
        ])
        self.nis: list[float] = []

    # ------------------------------------------------------------------ #
    def predict(self, gyro_x: float, gyro_z: float, dt: float,
                tilt_cmd: float = 0.0) -> None:
        """Propagate the autopilot model, using the gyro as a rate measurement."""
        tilt, rate, azim, bg = self.x
        wn, zeta = self.wn, self.zeta
        # autopilot: tilt_ddot = wn^2 (cmd - tilt) - 2 zeta wn tilt_dot
        tilt_acc = wn * wn * (tilt_cmd - tilt) - 2.0 * zeta * wn * rate
        self.x[0] = tilt + rate * dt + 0.5 * tilt_acc * dt * dt
        self.x[1] = rate + tilt_acc * dt
        # azimuth: held by the RCS, so the gyro-z rate is a direct measurement
        self.x[2] = azim + (gyro_z - 0.0) * dt

        qa = (self.cfg.att_process_std_deg * DEG) ** 2
        qb = (self.cfg.gyro_bias_rw_deg * DEG) ** 2
        # Jacobian of the autopilot model
        F = np.eye(4)
        F[0, 0] = 1.0 - 0.5 * wn * wn * dt * dt
        F[0, 1] = dt * (1.0 - zeta * wn * dt)
        F[1, 0] = -wn * wn * dt
        F[1, 1] = 1.0 - 2.0 * zeta * wn * dt
        F[2, 2] = 1.0
        Q = np.diag([qa * dt, qa * dt, qa * dt, qb * dt])
        self.P = _sym(F @ self.P @ F.T + Q)

        # ---- gyro-x as a measurement of tilt_rate + b_gyro ---------------
        H = np.zeros((1, 4)); H[0, 1] = 1.0; H[0, 3] = 1.0
        R = np.array([[(self.gyro_noise_deg * DEG) ** 2]])
        y = np.array([gyro_x]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ _safe_inv(S)
        self.x = self.x + K @ y
        self.P = _sym((np.eye(4) - K @ H) @ self.P)
        self.nis.append(float(y @ _safe_inv(S) @ y))

    # ------------------------------------------------------------------ #
    def update_azim_prior(self, azim: float, std_deg: float = 2.0) -> None:
        """Hold the azimuth to the commanded heading (the RCS maintains it)."""
        H = np.zeros((1, 4)); H[0, 2] = 1.0
        R = np.array([[(std_deg * DEG) ** 2]])
        y = np.array([azim]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ _safe_inv(S)
        self.x = self.x + K @ y
        self.P = _sym((np.eye(4) - K @ H) @ self.P)

    @property
    def tilt(self) -> float:
        return float(self.x[0])

    @property
    def tilt_rate(self) -> float:
        return float(self.x[1])

    @property
    def azim(self) -> float:
        return float(self.x[2])

    @property
    def gyro_bias(self) -> float:
        return float(self.x[3])


def _body_axis(tilt: float, azim: float) -> np.ndarray:
    st, ct = np.sin(tilt), np.cos(tilt)
    return np.array([st * np.cos(azim), ct, st * np.sin(azim)])


# --------------------------------------------------------------------------- #
#  Disturbance predictor  ("预测感知")
# --------------------------------------------------------------------------- #
class DisturbancePredictor:
    """Identifies the wind *profile* and forecasts the disturbance ahead.

    The navigation EKF supplies a lumped disturbance acceleration ``a_b`` which,
    on this vehicle, is dominated by aerodynamic drag:

        a_b = -0.5 k(h) |v - w| (v - w),      k(h) = rho(h) cd A / m

    Because ``v`` and ``k(h)`` are known, that vector equation inverts for the
    wind velocity at the current altitude (see ``update``).

    The forecast exploits the fact that **the wind is a function of altitude**:
    as the vehicle descends it samples ``w`` at progressively lower altitudes,
    which lets it fit a local profile ``w(h)`` and then evaluate that profile at
    the altitude it is *about to reach*.  This is what makes the forecast
    genuinely predictive rather than a blind extrapolation in time -- a shear
    layer below the vehicle can be anticipated before the vehicle flies into it.

    The profile is modelled as locally linear in altitude and tracked by
    recursive least squares, so it adapts as new layers are sampled.
    """

    def __init__(self, cfg: EstimatorConfig, horizon: float = 0.20,
                 n_steps: int = 4):
        self.cfg = cfg
        #: forecast window = one control period (50 ms) + the actuator lag
        #: (150 ms); the interval the command being computed will act over
        self.horizon = horizon
        self.n_steps = n_steps
        self.a_filt = np.zeros(3)
        self.wind_est = np.zeros(3)
        self.predicted = np.zeros(3)
        self.valid = False
        self._init = False

        # linear wind profile w(h) = w0 + w1 * (h - h_ref), per horizontal axis
        self._h_ref = 0.0
        self.w0 = np.zeros(3)
        self.w1 = np.zeros(3)
        self._P = np.eye(2) * 1.0e4        # RLS covariance per axis (shared)
        self._lam = 0.995                   # forgetting factor
        self._n_fit = 0

    # ------------------------------------------------------------------ #
    def update(self, a_meas: np.ndarray, vel: np.ndarray, altitude: float,
               rho_cdA_over_m: float, dt: float) -> None:
        """Low-pass the estimate, invert the drag law for the wind, and refit
        the altitude-dependent wind profile."""
        if not self._init:
            self.a_filt = np.asarray(a_meas, dtype=float).copy()
            self._h_ref = float(altitude)
            self._init = True
            return
        alpha = 1.0 - np.exp(-dt / max(self.cfg.dist_lpf_tau, 1e-6))
        self.a_filt = self.a_filt + alpha * (np.asarray(a_meas, dtype=float) - self.a_filt)

        a_mag = float(np.linalg.norm(self.a_filt))
        if a_mag < 1e-3 or rho_cdA_over_m < 1e-9:
            self.valid = False
            return
        # |a_b| = 0.5 k |rel|^2  =>  |rel| = sqrt(2 |a_b| / k)
        rel_mag = math.sqrt(2.0 * a_mag / rho_cdA_over_m)
        rel = -self.a_filt / a_mag * rel_mag  # a_b is antiparallel to rel
        w = np.asarray(vel, dtype=float) - rel
        beta = 1.0 - np.exp(-dt / 1.0)
        self.wind_est = self.wind_est + beta * (w - self.wind_est)
        self.valid = True

        # ---- refit the linear profile w(h) -------------------------------
        # Only the horizontal components are modelled: the mean wind is
        # horizontal and the vertical component is negligible.
        h = float(altitude) - self._h_ref
        phi = np.array([1.0, h])
        # skip the fit while the aero signal is too weak to identify the wind
        if rel_mag > 8.0:
            self._n_fit += 1
            for k in (0, 2):  # x and z
                P = self._P
                denom = self._lam + phi @ P @ phi
                K = (P @ phi) / denom
                P_new = (P - np.outer(K, phi @ P)) / self._lam
                err = float(w[k]) - float(phi @ np.array([self.w0[k], self.w1[k]]))
                coef = np.array([self.w0[k], self.w1[k]]) + K * err
                self.w0[k], self.w1[k] = float(coef[0]), float(coef[1])
                self._P = P_new
            # a plausible wind profile has a bounded speed
            self.w0 = np.clip(self.w0, -60.0, 60.0)
            self.w1 = np.clip(self.w1, -0.35, 0.35)

    # ------------------------------------------------------------------ #
    def wind_at(self, altitude: float) -> np.ndarray:
        """Evaluate the fitted wind profile at an altitude."""
        if self._n_fit < 6:
            return self.wind_est.copy()
        h = float(altitude) - self._h_ref
        w = self.w0 + self.w1 * h
        # never report a wind wildly different from the last direct estimate
        delta = w - self.wind_est
        n = float(np.linalg.norm(delta))
        if n > 25.0:
            w = self.wind_est + delta / n * 25.0
        return w

    # ------------------------------------------------------------------ #
    def predict_horizon(self, pos: np.ndarray, vel: np.ndarray,
                        a_cmd: np.ndarray, rho_cdA_over_m_fn,
                        gravity: float = G0) -> np.ndarray:
        """Forecast the mean disturbance over the control window.

        The forward model flies the *baseline command* and, crucially, samples
        the wind from the fitted **altitude profile** rather than assuming it
        stays constant.  A shear layer below the vehicle is therefore seen
        before the vehicle reaches it.
        """
        if not self.valid:
            self.predicted = self.a_filt.copy()
            return self.predicted

        dt = self.horizon / self.n_steps
        p = np.asarray(pos, dtype=float).copy()
        v = np.asarray(vel, dtype=float).copy()
        acc = self.a_filt.copy()
        total = np.zeros(3)
        for _ in range(self.n_steps):
            a_total = np.asarray(a_cmd, dtype=float) + acc
            a_total[1] -= gravity
            v = v + a_total * dt
            p = p + v * dt
            w = self.wind_at(float(p[1]))
            rel = v - w
            k = rho_cdA_over_m_fn(float(p[1]))
            acc = -0.5 * k * float(np.linalg.norm(rel)) * rel
            total = total + acc
        self.predicted = total / self.n_steps
        return self.predicted

    @property
    def estimate(self) -> np.ndarray:
        return self.a_filt


# --------------------------------------------------------------------------- #
#  Bundled perception stack
# --------------------------------------------------------------------------- #
@dataclass
class PerceptionOutput:
    pos: np.ndarray
    vel: np.ndarray
    tilt: float
    azim: float
    disturbance: np.ndarray
    disturbance_pred: np.ndarray
    pos_std: np.ndarray
    vel_std: np.ndarray


class PerceptionStack:
    """Convenience wrapper running all three estimators together."""

    def __init__(self, cfg: Config, dt: float = 0.01):
        self.cfg = cfg
        self.nav = NavigationEKF(cfg.estimator, dt)
        self.att = AttitudeEKF(cfg.estimator, dt,
                               wn=cfg.rocket.att_wn, zeta=cfg.rocket.att_zeta,
                               gyro_noise_deg=cfg.sensors.gyro_noise_deg)
        self.pred = DisturbancePredictor(cfg.estimator)

    def initialise(self, pos: np.ndarray, vel: np.ndarray, tilt: float,
                   azim: float) -> None:
        self.nav.x[0:3] = pos
        self.nav.x[3:6] = vel
        self.att.x[0] = tilt
        self.att.x[2] = azim

    def output(self) -> PerceptionOutput:
        return PerceptionOutput(
            pos=self.nav.pos.copy(),
            vel=self.nav.vel.copy(),
            tilt=self.att.tilt,
            azim=self.att.azim,
            disturbance=self.nav.disturbance.copy(),
            disturbance_pred=self.pred.predicted.copy(),
            pos_std=self.nav.pos_std(),
            vel_std=self.nav.vel_std(),
        )
