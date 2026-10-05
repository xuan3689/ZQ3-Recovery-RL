"""Reinforcement-learning environment for residual recovery guidance.

The environment is a *residual* MDP: the action is not the throttle and gimbal
angle but a bounded acceleration correction layered on top of the analytic
powered-descent baseline.  This is what makes the problem learnable in minutes
on a CPU instead of hours on a GPU:

* the baseline already flies a feasible trajectory, so the reward is dense and
  the policy starts from a competent prior;
* the residual is bounded to ``[-2, 2] m/s^2`` per axis, so the policy can
  never command a divergent trajectory on its own;
* the safety shield projects the total command onto the admissible set, so
  constraint satisfaction is guaranteed by construction.

Observations are built exclusively from *estimated* quantities, which is what
closes the loop between the perception stack and the policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import G0, Config, air_density
from .dynamics import (
    ActuatorCommand,
    FlightState,
    VehicleDynamics,
    WindField,
    touchdown_metrics,
)
from .estimator import PerceptionStack
from .guidance import GuidanceController
from .sensors import SensorSuite

DEG = np.pi / 180.0


@dataclass
class StepInfo:
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    phase: str = ""
    tgo: float = 0.0
    fuel_used: float = 0.0
    shield_clip: bool = False
    lateral: float = 0.0
    speed: float = 0.0
    tilt_deg: float = 0.0
    est_pos_err: float = 0.0
    est_vel_err: float = 0.0


class RecoveryEnv:
    """Gym-style environment (no Gym dependency; a plain ``reset``/``step``)."""

    #: observation layout, in order.  The dimension is derived from this list
    #: so it can never drift out of sync with ``_observation``.
    OBS_FIELDS = (
        "dp_x", "dp_y", "dp_z",            # position error to the pad
        "dv_x", "dv_y", "dv_z",            # velocity error to the target
        "dv_lat",                          # lateral speed magnitude
        "tilt", "azim_sin", "azim_cos",    # attitude
        "dist_x", "dist_y", "dist_z",      # instantaneous disturbance estimate
        "pred_x", "pred_y", "pred_z",      # horizon disturbance forecast
        "wind_x", "wind_z",                # identified wind
        "fuel_frac",                       # remaining propellant
        "accel_max", "accel_min",          # current thrust envelope
        "tgo", "feasible",                 # guidance horizon & feasibility
        "prev_a_x", "prev_a_y", "prev_a_z",  # previous residual action
        "pos_std", "vel_std",              # EKF uncertainty
        "thrust_deficit",                  # EKF-observed thrust shortfall
    )
    OBS_DIM = len(OBS_FIELDS)

    def __init__(self, cfg: Optional[Config] = None, seed: int = 0,
                 curriculum_level: float = 1.0,
                 use_prediction: bool = True,
                 use_residual: bool = True,
                 use_sensors: bool = True):
        self.cfg = cfg or Config()
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.curriculum = float(np.clip(curriculum_level, 0.0, 1.0))
        self.use_prediction = use_prediction
        self.use_residual = use_residual
        self.use_sensors = use_sensors

        self.dyn = VehicleDynamics(self.cfg.rocket, self.cfg.mission, self.cfg)
        #: per-episode delivered-to-rated thrust ratio; drawn in ``reset``
        self.thrust_scale = 1.0
        self.controller = GuidanceController(
            self.cfg,
            use_prediction=use_prediction,
        )
        self.sensors = SensorSuite(self.cfg.sensors, self.rng)
        self.perception = PerceptionStack(self.cfg, self.cfg.env.dt_phys)

        self.rocket = self.cfg.rocket
        self.mission = self.cfg.mission
        self.env_cfg = self.cfg.env

        self.state = FlightState()
        self.wind = WindField(self.cfg.wind, self.rng)
        self.t = 0.0
        self._episode = 0
        self._last_cmd = ActuatorCommand()
        self._prev_action = np.zeros(3)
        self._tgo = 0.0
        self._fuel0 = 0.0
        self._shield_clip = False
        self._initialised = False

    # ------------------------------------------------------------------ #
    #  Curriculum
    # ------------------------------------------------------------------ #
    def _curriculum_scale(self) -> Tuple[float, float]:
        """Return (wind_scale, offset_scale) for the current curriculum level.

        Level 0 is a no-wind, no-offset drop straight onto the pad; level 1 is
        the full envelope.  Both scales grow monotonically so that training
        never has to un-learn a solved skill.
        """
        c = self.curriculum
        wind_scale = float(np.clip(c, 0.05, 1.0))
        offset_scale = float(np.clip(c, 0.15, 1.0))
        return wind_scale, offset_scale

    # ------------------------------------------------------------------ #
    #  Reset
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._episode += 1
        ws, os_ = self._curriculum_scale()
        m = self.mission
        rng = self.rng

        x0 = float(rng.uniform(*m.x0_range)) * os_
        z0 = float(rng.uniform(*m.z0_range)) * os_
        alt0 = m.ignite_alt + float(rng.uniform(-150.0, 150.0)) * os_
        vx0 = float(rng.uniform(*m.v0_x_range)) * os_
        vz0 = float(rng.uniform(*m.v0_z_range)) * os_
        vy0 = m.v0_y * float(rng.uniform(0.92, 1.08))

        self.state = FlightState(
            pos=np.array([x0, alt0, z0]),
            vel=np.array([vx0, vy0, vz0]),
            mass=self.rocket.m0,
            tilt=float(rng.uniform(-2.0, 2.0) * DEG) * os_,
            tilt_rate=0.0,
            azim=float(rng.uniform(-np.pi, np.pi)),
            throttle=0.0,
            engine_on=True,
        )
        self._fuel0 = self.rocket.m0

        # thrust degradation: drawn per episode, unknown to the guidance law.
        # Scaled by the curriculum so the easy stages stay nominal.
        lo, hi = self.rocket.thrust_scale_range
        c = self.curriculum
        self.thrust_scale = float(1.0 - (1.0 - rng.uniform(lo, hi)) * c)
        self.dyn.thrust_scale = self.thrust_scale

        # wind: scale the base speed by the curriculum
        self.wind = WindField(self.cfg.wind, rng)
        if self.cfg.wind.enable:
            self.wind.speed *= ws
            self.wind.turb = rng.normal(0.0, self.cfg.wind.turb_sigma * ws, 3)

        self.sensors.reset()
        self.perception = PerceptionStack(self.cfg, self.env_cfg.dt_phys)
        self.perception.initialise(self.state.pos, self.state.vel,
                                   self.state.tilt, self.state.azim)

        self.t = 0.0
        self._prev_action = np.zeros(3)
        self._tgo = 0.0
        self._shield_clip = False
        self._initialised = True
        self._phi_prev = self._potential(self.perception.output())
        return self._observation(), {"episode": self._episode}

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #
    def _observation(self) -> np.ndarray:
        """28-dim observation assembled from the *estimated* state.

        When the ablation disables prediction, the forecast features are zeroed
        so the policy genuinely lacks the predictive channel rather than merely
        having it zero-weighted.
        """
        cfg = self.cfg
        out = self.perception.output()
        pos, vel = out.pos, out.vel
        target = np.asarray(self.mission.target, dtype=float)

        dp = pos - target
        dv = vel - np.array([0.0, self.mission.v_target_y, 0.0])
        accel_min, accel_max = self.dyn.available_accel(self.state)
        sol = self.controller.last_solution
        tgo = sol.tgo if sol is not None else 0.0
        pred = out.disturbance_pred if self.use_prediction else np.zeros(3)

        feats = [
            dp[0] / 400.0, dp[1] / 1000.0, dp[2] / 400.0,
            dv[0] / 60.0, dv[1] / 100.0, dv[2] / 60.0,
            float(np.linalg.norm(dv[0::2])) / 60.0,
            out.tilt / (25.0 * DEG),
            np.sin(out.azim), np.cos(out.azim),
            out.disturbance[0] / 5.0, out.disturbance[1] / 5.0,
            out.disturbance[2] / 5.0,
            pred[0] / 5.0, pred[1] / 5.0, pred[2] / 5.0,
            # the identified wind and the forecast are the *predictive
            # perception* channel; zeroing them when the ablation disables
            # prediction makes the comparison meaningful
            (self.perception.pred.wind_est[0] / 25.0) if self.use_prediction else 0.0,
            (self.perception.pred.wind_est[2] / 25.0) if self.use_prediction else 0.0,
            (self.state.mass - self.rocket.m_dry) / self.rocket.fuel_mass,
            accel_max / 30.0,
            accel_min / 30.0,
            tgo / 30.0,
            1.0 if sol is not None and sol.feasible else 0.0,
            self._prev_action[0] / cfg.guidance.residual_max,
            self._prev_action[1] / cfg.guidance.residual_max,
            self._prev_action[2] / cfg.guidance.residual_max,
            float(np.clip(out.pos_std[0] / 3.0, 0.0, 3.0)),
            float(np.clip(out.vel_std[0] / 1.0, 0.0, 3.0)),
            # thrust shortfall along the body axis, estimated from the
            # accelerometer: a degraded engine shows up as the delivered
            # acceleration being smaller than commanded.  This is the feature
            # that lets the policy anticipate an under-performing engine instead
            # of merely reacting to the resulting altitude loss.
            float(np.clip(
                (self._last_cmd.throttle * accel_max
                 - float(np.dot(out.disturbance, -self.state.body_axis())))
                / max(accel_max, 1e-6), -1.0, 1.0)),
        ]
        obs = np.asarray(feats, dtype=np.float32)
        assert obs.shape[0] == self.OBS_DIM, (
            f"observation length {obs.shape[0]} != OBS_DIM {self.OBS_DIM}")
        return np.clip(obs, -cfg.env.obs_clip, cfg.env.obs_clip)

    # ------------------------------------------------------------------ #
    #  Step
    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        cfg = self.cfg
        ecfg = self.env_cfg
        gcfg = cfg.guidance

        action = np.asarray(action, dtype=float).reshape(3)
        if self.use_residual:
            residual = np.clip(action, -1.0, 1.0) * gcfg.residual_max
        else:
            residual = np.zeros(3)

        reward = 0.0
        info = StepInfo()
        n_sub = max(int(round(ecfg.dt_ctrl / ecfg.dt_phys)), 1)
        dt = ecfg.dt_phys

        for _ in range(n_sub):
            # ---- perception -------------------------------------------
            fresh = self.sensors.step(self.t, self.state,
                                      self.wind.velocity(self.state.pos[1]),
                                      self.state.throttle, self.rocket,
                                      self.thrust_scale)
            accel_min, accel_max = self.dyn.available_accel(self.state)

            if self.use_sensors:
                from .dynamics import body_from_world
                # The estimator must never see the ground-truth attitude, so the
                # commanded thrust direction is reconstructed from the
                # *estimated* tilt/azimuth carried by the attitude filter.
                est_tilt = self.perception.att.tilt
                est_azim = self.perception.att.azim
                cmd_thrust_acc = self._last_cmd.throttle * accel_max * \
                    body_from_world(est_tilt, est_azim)[0]
                # specific force in the world frame, reconstructed through the
                # estimated attitude (gravity is excluded by the definition of
                # a specific-force measurement)
                R_est = body_from_world(est_tilt, est_azim)
                sf_world = R_est.T @ self.sensors.imu.accel

                if fresh["imu"]:
                    # propagate with the known command (thrust + gravity) ...
                    self.perception.nav.predict(
                        cmd_thrust_acc - np.array([0.0, G0, 0.0]), dt)
                    # ... and fold the accelerometer into the disturbance state
                    self.perception.nav.update_accel(sf_world, cmd_thrust_acc)
                    # attitude: propagate the autopilot model with the command,
                    # using the gyro as a rate measurement
                    self.perception.att.predict(self.sensors.imu.gyro[0],
                                                self.sensors.imu.gyro[2], dt,
                                                self._last_cmd.tilt)
                    self.perception.att.update_azim_prior(
                        self._last_cmd.azim,
                        self.cfg.estimator.azim_prior_std_deg)
                    out_tmp = self.perception.output()
                    self.perception.pred.update(
                        out_tmp.disturbance, out_tmp.vel, out_tmp.pos[1],
                        float(air_density(out_tmp.pos[1])) * self.rocket.cd_a
                        / self.state.mass, dt)
                if fresh["gnss"]:
                    self.perception.nav.update_gnss(self.sensors.gnss.pos,
                                                    self.sensors.gnss.vel, self.t)
                if fresh["baro"]:
                    self.perception.nav.update_baro(self.sensors.baro.altitude, self.t)
            else:
                self.perception.initialise(self.state.pos, self.state.vel,
                                           self.state.tilt, self.state.azim)

            out = self.perception.output()

            # ---- stage 1: analytic baseline command --------------------
            sol = self.controller.compute_baseline(out.pos, out.vel,
                                                  accel_min, accel_max)

            # ---- predictive disturbance forecast ----------------------
            # built from the *baseline* command so the forward model follows
            # the trajectory the vehicle will actually fly
            if self.use_prediction:
                k_fn = lambda h: float(air_density(h)) * self.rocket.cd_a \
                    / self.state.mass
                pred = self.perception.pred.predict_horizon(
                    out.pos, out.vel, sol.accel, k_fn, G0)
            else:
                pred = np.zeros(3)

            # ---- stage 2: feed-forward + residual + safety shield ------
            accel, cmd, rep = self.controller.finalize(
                sol, pred, residual, accel_min, accel_max, out.pos)
            self._tgo = sol.tgo
            self._shield_clip = rep.clipped_throttle or rep.clipped_tilt
            self._last_cmd = cmd

            # ---- plant ------------------------------------------------
            self.state = self.dyn.step(
                self.state, cmd, self.wind.velocity(self.state.pos[1]),
                dt, self.rocket.throttle_tau,
            )
            self.wind.advance(self.t, dt, self.rng)
            self.t += dt

            # ---- shaping reward (telescoping, safe at 100 Hz) ----------
            reward += self._step_reward(out)

            # ---- termination ------------------------------------------
            done, trunc, info = self._check_termination(out)
            if done or trunc:
                break

        # ---- control-effort penalties (once per 20 Hz control step) -----
        # kept out of the substep loop so they are not multiplied by the
        # substep ratio -- see ``_effort_penalty``
        reward += self._effort_penalty(out, residual)

        # terminal reward -- only when the episode actually ended
        if info.terminated or info.truncated:
            reward += self._terminal_reward(info)
        self._prev_action = action.copy()
        obs = self._observation()
        info.fuel_used = self._fuel0 - self.state.mass
        info.tgo = self._tgo
        info.shield_clip = self._shield_clip
        return obs, float(reward), info.terminated, info.truncated, {
            "success": info.success, "phase": info.phase,
            "lateral": info.lateral, "speed": info.speed,
            "tilt_deg": info.tilt_deg, "fuel_used": info.fuel_used,
            "tgo": info.tgo, "shield_clip": info.shield_clip,
            "est_pos_err": info.est_pos_err, "est_vel_err": info.est_vel_err,
        }

    # ------------------------------------------------------------------ #
    def _potential(self, out) -> float:
        """Landing-quality potential ``Phi(s) <= 0`` (0 only at a perfect
        touchdown).

        The errors are normalised by the **success tolerances** (3 m lateral,
        2.5 m/s) rather than by the envelope size, and the growth is
        logarithmic:

            Phi = -w * ln(1 + (d/d_tol)^2 + (|v - v_t|/v_tol)^2)

        Two properties matter here.

        *Steepness where it counts.* Normalising by the tolerance puts the
        success boundary at ``Phi = -w ln 2`` and gives a gradient of order
        ``w / d_tol`` right at the pad. An earlier version normalised by the
        envelope (1000 m), which left a gradient of ~1e-4 per metre near
        touchdown -- the policy received essentially no signal about the 3 m
        boundary, so the sparse terminal bonus dominated and the residual
        learned a bias instead of a control law.

        *Bounded growth.* The logarithm keeps the far-field penalty small, so
        the episode return stays O(100) instead of exploding with the initial
        offset (a quadratic in ``d/d_tol`` would reach 1e6 at 600 m).
        """
        ecfg = self.env_cfg
        m = self.mission
        target = np.asarray(m.target, dtype=float)
        d = float(np.linalg.norm((out.pos - target)[[0, 2]]))
        v_err = float(np.linalg.norm(out.vel - np.array([0.0, m.v_target_y, 0.0])))
        r2 = (d / m.pos_tol) ** 2 + (v_err / m.vel_tol) ** 2
        return -ecfg.w_quality * math.log1p(r2)

    def _step_reward(self, out) -> float:
        """Potential-based shaping, evaluated once per *physics* substep.

        ``r = Phi(s') - Phi(s)``

        The shaping term telescopes over the substeps, so the episode total is
        ``Phi(s_final) - Phi(s_0)`` regardless of the substep count -- which is
        what makes it safe to evaluate at 100 Hz while the *effort* penalties
        (see :meth:`_effort_penalty`) are applied once per 20 Hz control step.

        Splitting the two is essential.  An earlier version folded the effort
        penalties into this per-substep function, which multiplied them by the
        substep ratio (5x): the residual penalty alone reached ~2040 per episode
        against a success bonus of 100, so the optimal policy was to output a
        zero residual -- the learned controller degenerated exactly into the
        analytic baseline.
        """
        phi_next = self._potential(out)
        r = phi_next - self._phi_prev
        self._phi_prev = phi_next
        return float(r)

    def _effort_penalty(self, out, residual: np.ndarray) -> float:
        """Control-effort penalties, applied once per control step.

        ``r = -c_tilt * tilt_n^2 - c_res * |a_rl|^2``

        Both terms are normalised so a *sustained* worst-case effort costs far
        less than the success bonus.  At full tilt the attitude term costs
        ``c_tilt`` per step, and at the full residual bound the action term costs
        ``c_res`` per step; over a ~510-step episode that is a few tens of points
        each, comfortably below the success/failure spread (``r_success = 100``,
        the failure penalty is graded down to ``-100``).  The penalties must stay
        well below that spread, otherwise the policy is rewarded for doing
        nothing -- which is exactly how an earlier, 5x-inflated version
        degenerated into the analytic baseline.
        """
        ecfg = self.env_cfg
        # attitude effort, normalised by the cruise authority so the penalty is
        # O(c_tilt) at full tilt rather than O(c_tilt * 625) (a degrees/radians
        # trap)
        tilt_n = out.tilt / (25.0 * DEG)
        # residual effort, normalised by the action bound so the penalty is
        # O(c_res) at full authority rather than O(c_res * a_max_res^2)
        res_n = np.asarray(residual, dtype=float) / max(
            self.cfg.guidance.residual_max, 1e-9)
        r = -ecfg.w_tilt * tilt_n * tilt_n
        r -= ecfg.w_residual * float(np.dot(res_n, res_n))
        return float(r)

    # ------------------------------------------------------------------ #
    def _terminal_reward(self, info: StepInfo) -> float:
        """Terminal bonus/penalty on top of the telescoped shaping term.

        The potential-based shaping already accounts for *how close* the final
        state is to a good landing, so the terminal term only has to encode the
        discrete success/failure distinction and the residual accuracy.
        """
        ecfg = self.env_cfg
        if info.success:
            return ecfg.r_success
        if info.phase == "crash":
            # graded penalty: a near-miss is much better than a 300 m miss
            miss = min(info.lateral / 50.0, 1.0)
            speed = min(abs(info.speed) / 20.0, 1.0)
            return ecfg.r_crash * (0.4 + 0.3 * miss + 0.3 * speed)
        # timeout: penalise proportional to the miss
        return ecfg.r_crash * 0.3 * min(info.lateral / 50.0 + 0.5, 1.0)

    # ------------------------------------------------------------------ #
    def _check_termination(self, out) -> Tuple[bool, bool, StepInfo]:
        m = self.mission
        info = StepInfo()
        info.est_pos_err = float(np.linalg.norm(out.pos - self.state.pos))
        info.est_vel_err = float(np.linalg.norm(out.vel - self.state.vel))

        alt = self.state.pos[1]
        # hard failure: below the pad, or absurd attitude / lateral excursion
        if alt <= m.target[1] - 0.5:
            met = touchdown_metrics(self.state, m)
            info.lateral = met["lateral"]
            info.speed = met["speed"]
            info.tilt_deg = met["tilt_deg"]
            info.success = met["success"]
            info.phase = "landed" if info.success else "crash"
            info.terminated = True
            return True, False, info

        lateral = float(np.hypot(self.state.pos[0] - m.target[0],
                                 self.state.pos[2] - m.target[2]))
        if lateral > 1200.0 or abs(self.state.tilt) > 60.0 * DEG:
            met = touchdown_metrics(self.state, m)
            info.lateral = met["lateral"]
            info.speed = met["speed"]
            info.tilt_deg = met["tilt_deg"]
            info.phase = "crash"
            info.terminated = True
            return True, False, info

        if self.t >= m.max_time:
            met = touchdown_metrics(self.state, m)
            info.lateral = met["lateral"]
            info.speed = met["speed"]
            info.tilt_deg = met["tilt_deg"]
            info.phase = "timeout"
            info.truncated = True
            return False, True, info

        info.phase = self._phase(alt)
        return False, False, info

    @staticmethod
    def _phase(alt: float) -> str:
        if alt > 1000.0:
            return "high_decel"
        if alt > 600.0:
            return "mid_decel"
        if alt > 300.0:
            return "approach"
        if alt > 100.0:
            return "final_approach"
        if alt > 30.0:
            return "terminal"
        return "touchdown"

    # ------------------------------------------------------------------ #
    #  Introspection for plotting
    # ------------------------------------------------------------------ #
    def rollout_trace(self, policy=None) -> Dict[str, np.ndarray]:
        """Run one episode and record everything needed for the figures."""
        obs, _ = self.reset()
        trace: Dict[str, list] = {k: [] for k in (
            "t", "pos", "vel", "tilt_deg", "throttle", "mass",
            "est_pos", "est_vel", "est_tilt_deg", "dist_est", "dist_pred",
            "true_dist", "reward", "phase", "shield_clip", "pos_std")}
        done = trunc = False
        total_r = 0.0
        while not (done or trunc):
            if policy is not None:
                action = policy(obs)
            else:
                action = np.zeros(3)
            obs, r, done, trunc, info = self.step(action)
            total_r += r
            out = self.perception.output()
            wind_v = self.wind.velocity(self.state.pos[1])
            # the true un-modelled acceleration: aero + wind mismatch
            true_dist = -0.5 * float(np.exp(-max(self.state.pos[1], 0) / 8500.0)
                                     * 1.225) * np.linalg.norm(self.state.vel - wind_v) \
                * (self.state.vel - wind_v) * self.rocket.cd_a / self.state.mass
            trace["t"].append(self.t)
            trace["pos"].append(self.state.pos.copy())
            trace["vel"].append(self.state.vel.copy())
            trace["tilt_deg"].append(abs(self.state.tilt) / DEG)
            trace["throttle"].append(self.state.throttle)
            trace["mass"].append(self.state.mass)
            trace["est_pos"].append(out.pos.copy())
            trace["est_vel"].append(out.vel.copy())
            trace["est_tilt_deg"].append(abs(out.tilt) / DEG)
            trace["dist_est"].append(out.disturbance.copy())
            trace["dist_pred"].append(out.disturbance_pred.copy())
            trace["true_dist"].append(true_dist.copy())
            trace["reward"].append(r)
            trace["phase"].append(info["phase"])
            trace["shield_clip"].append(float(info["shield_clip"]))
            trace["pos_std"].append(out.pos_std.copy())
        out_dict = {k: np.asarray(v) for k, v in trace.items()}
        out_dict["total_reward"] = np.array([total_r])
        out_dict["success"] = np.array([bool(info["success"])])
        out_dict["lateral"] = np.array([info["lateral"]])
        out_dict["speed"] = np.array([info["speed"]])
        out_dict["tilt_deg_final"] = np.array([info["tilt_deg"]])
        out_dict["fuel_used"] = np.array([info["fuel_used"]])
        return out_dict
