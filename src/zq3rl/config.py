"""Configuration objects for the Zhuque-3 (ZQ3) vertical-recovery project.

All physical quantities are SI unless the field name says otherwise.
The numbers follow the public parameters used in the course lab manual
(``实验指导书（一）``) and the ``ZQ3-Recovery-Sim`` reference implementation
(``实验指导书（二）``), scaled so that a full episode runs in a few seconds of
CPU time.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

G0 = 9.80665  # standard gravity [m/s^2]
RHO0 = 1.225  # sea-level air density [kg/m^3]
SCALE_HEIGHT = 8500.0  # exponential-atmosphere scale height [m]


# --------------------------------------------------------------------------- #
#  Rocket
# --------------------------------------------------------------------------- #
@dataclass
class RocketConfig:
    """Physical parameters of the reusable first stage."""

    m0: float = 30000.0  # wet mass at ignition [kg]
    m_dry: float = 25500.0  # dry mass (structure + residual) [kg]
    isp: float = 330.0  # specific impulse [s]
    thrust_max_acc0: float = 0.0  # explicit override; 0 -> use ``twr0``
    throttle_min: float = 0.35  # deepest sustainable throttle [-]
    throttle_max: float = 1.00
    throttle_tau: float = 0.15  # engine first-order response time constant [s]
    diameter: float = 4.5  # reference diameter [m]
    cd: float = 0.85  # drag coefficient (constant, subsonic lumped) [-]

    tilt_max_deg: float = 25.0  # attitude authority during descent [deg]
    tilt_max_terminal_deg: float = 8.0  # attitude authority near touchdown [deg]
    tilt_rate_max_deg: float = 30.0  # gimbal / RCS slew limit [deg/s]
    att_wn: float = 3.5  # attitude autopilot natural frequency [rad/s]
    att_zeta: float = 0.90  # attitude autopilot damping [-]

    #: thrust-to-weight at ignition.  2.2 leaves ~4.5 m/s^2 of lateral
    #: authority at a 25 deg tilt while arresting the descent, i.e. the envelope
    #: is feasible for the analytic baseline in still air and it is the
    #: *disturbance* that breaks it -- which is where prediction and learning
    #: pay off.  Sweeping 1.8/2.2/2.6/3.0 gives 65%/80%/0%/0% in still air.
    twr0: float = 2.20

    #: Per-episode thrust degradation: the *delivered* thrust is
    #: ``scale * rated``, with ``scale`` drawn uniformly from this range.  Real
    #: engines deliver less than their rated thrust as they age, and a
    #: multi-engine vehicle must cope with one engine running weak.  Crucially
    #: the guidance law plans against the **rated** envelope, so it does not
    #: know the scale: the discrepancy is observable only through the
    #: accelerometer, i.e. exactly the channel the navigation EKF's disturbance
    #: state captures.  This is the off-nominal condition that separates a
    #: feedback-only controller (which reacts after the trajectory has already
    #: drifted) from the invention (which anticipates it).
    thrust_scale_range: Tuple[float, float] = (0.82, 1.00)

    leg_height: float = 6.0  # body-frame height of the landing legs [m]

    # ----- derived ----------------------------------------------------------
    @property
    def area(self) -> float:
        return math.pi * (self.diameter / 2.0) ** 2

    @property
    def cd_a(self) -> float:
        return self.cd * self.area

    @property
    def thrust_max(self) -> float:
        # thrust-to-weight at ignition sets the envelope; ``thrust_max_acc0``
        # is kept as an explicit override for ablation runs
        a0 = self.thrust_max_acc0 if self.thrust_max_acc0 > 0 else self.twr0 * G0
        return a0 * self.m0

    @property
    def exhaust_velocity(self) -> float:
        return self.isp * G0

    @property
    def fuel_mass(self) -> float:
        return self.m0 - self.m_dry


# --------------------------------------------------------------------------- #
#  Mission
# --------------------------------------------------------------------------- #
@dataclass
class MissionConfig:
    """Initial condition envelope and landing target."""

    alt0: float = 2500.0  # altitude at the start of the entry coast [m]
    ignite_alt: float = 1600.0  # engine re-ignition altitude [m]
    target: Tuple[float, float, float] = (0.0, 6.0, 0.0)  # landing point [m]
    v_target_y: float = -1.5  # touchdown sink rate [m/s]
    v0_y: float = -150.0  # initial vertical speed [m/s]
    x0_range: Tuple[float, float] = (100.0, 650.0)  # downrange offset [m]
    z0_range: Tuple[float, float] = (-200.0, 200.0)  # crossrange offset [m]
    v0_x_range: Tuple[float, float] = (-42.0, -12.0)  # initial downrange speed
    v0_z_range: Tuple[float, float] = (-18.0, 18.0)

    # success / failure criteria evaluated at touchdown
    pos_tol: float = 3.0  # lateral landing accuracy [m]
    vel_tol: float = 2.5  # touchdown speed [m/s]
    tilt_tol_deg: float = 8.0  # touchdown tilt [deg]

    max_time: float = 70.0  # episode time limit [s]


# --------------------------------------------------------------------------- #
#  Atmosphere / wind
# --------------------------------------------------------------------------- #
@dataclass
class WindConfig:
    """Randomised wind field: mean profile + shear layer + turbulence.

    The envelope is deliberately severe: the mean wind reaches 22 m/s with a
    sharp low-altitude shear layer and correlated gusts.  This is what pushes
    the analytic baseline off the pad and gives the residual policy something
    to correct.
    """

    base_speed_range: Tuple[float, float] = (0.0, 22.0)  # [m/s]
    direction_range: Tuple[float, float] = (0.0, 2.0 * math.pi)  # [rad]
    shear_alt: float = 350.0  # altitude of the shear layer [m]
    shear_gain: float = 0.85  # fractional speed jump across the layer [-]
    turb_sigma: float = 3.2  # turbulence std [m/s]
    turb_tau: float = 2.5  # turbulence correlation time [s]
    enable: bool = True


# --------------------------------------------------------------------------- #
#  Sensors
# --------------------------------------------------------------------------- #
@dataclass
class SensorConfig:
    imu_rate: float = 100.0  # [Hz]
    gnss_rate: float = 10.0
    baro_rate: float = 20.0

    accel_noise: float = 0.030  # white noise std [m/s^2]
    accel_bias0: float = 0.060  # initial bias magnitude [m/s^2]
    accel_bias_rw: float = 0.004  # bias random-walk [m/s^2/sqrt(s)]
    gyro_noise_deg: float = 0.05  # white noise std [deg/s]
    gyro_bias0_deg: float = 0.30  # initial gyro bias [deg/s]
    gyro_bias_rw_deg: float = 0.010

    gnss_pos_noise: float = 1.20  # [m]
    gnss_vel_noise: float = 0.25  # [m/s]
    baro_noise: float = 0.35  # [m]

    enable: bool = True


# --------------------------------------------------------------------------- #
#  Estimator
# --------------------------------------------------------------------------- #
@dataclass
class EstimatorConfig:
    # navigation EKF (position, velocity, lumped disturbance acceleration)
    pos_init_std: float = 3.0
    vel_init_std: float = 1.5
    bias_init_std: float = 0.5
    accel_noise_std: float = 0.12  # process noise on velocity [m/s^2]
    bias_rw_std: float = 0.10  # process noise on the disturbance state
    gnss_pos_std: float = 1.30
    gnss_vel_std: float = 0.28
    baro_std: float = 0.40

    # attitude EKF (tilt, azimuth, two gyro biases)
    att_init_std_deg: float = 1.0
    gyro_bias_init_std_deg: float = 0.5
    att_process_std_deg: float = 0.05
    gyro_bias_rw_deg: float = 0.01
    aero_dir_std_deg: float = 4.0
    aero_min_accel: float = 0.35  # only update when aero dominates [m/s^2]
    use_aero_tilt_update: bool = False  # accelerometer-based tilt aid
    tilt_prior_std_deg: float = 1.20  # soft measurement of the tilt command
    azim_prior_std_deg: float = 2.0

    # disturbance observer
    dist_lpf_tau: float = 0.25  # low-pass time constant [s]

    enable_nav: bool = True
    enable_attitude: bool = True
    enable_disturbance: bool = True


# --------------------------------------------------------------------------- #
#  Guidance
# --------------------------------------------------------------------------- #
@dataclass
class GuidanceConfig:
    """Powered-descent guidance + PD tracking + residual RL."""

    # Apollo-style polynomial (powered descent) guidance
    tgo_min: float = 6.0  # lower bound on time-to-go [s]
    tgo_margin: float = 1.06  # inflate the heuristic time-to-go
    tgo_safety: float = 1.30  # inflate again if the command saturates
    accel_limit_frac: float = 0.92  # fraction of the available thrust used to
    #                                 estimate the feasible time-to-go

    # tracking gains (only used when the polynomial guidance is disabled)
    kp_pos: float = 0.35
    kd_vel: float = 1.05

    # residual action bounds.
    #
    # The bound is a *design* choice, not a detail: it sets how much authority
    # the learned policy has to correct what the analytic baseline gets wrong.
    # At 2 m/s^2 the residual is smaller than the baseline's own tracking error
    # over the terminal phase, so the policy can only add noise; a larger bound
    # gives it leverage to bias the terminal lateral correction while staying
    # small against the ~22 m/s^2 thrust envelope.
    #
    # Measured, though, 4 m/s^2 is *too much* authority.  A residual-authority
    # sweep on a trained policy (100 paired episodes, identical seeds) gives
    # 58% success at zero residual, 64% at 0.36 m/s^2, 69% at 0.72, 67% at 1.08,
    # and 45% at the full 1.44 m/s^2 the policy actually commands -- it wins 12
    # episodes and loses 1 against the baseline at half authority, but loses 27
    # and wins 14 at full.  The reason is structural: the telescoped shaping
    # reward depends only on the *mean* terminal error, so it keeps paying for
    # more residual until the mean stops improving, while the success criterion
    # is a *threshold* (3 m / 2.5 m/s) that a higher-variance trajectory fails
    # even at a lower mean.  Capping the bound at 1 m/s^2 keeps the learned
    # residual inside the helpful band -- the policy saturates instead of
    # over-commanding -- and leaves the shield's feasibility guarantee intact.
    residual_max: float = 1.0  # [m/s^2]

    # Feed-forward of the *predicted* disturbance.
    #
    # Measured on this vehicle and envelope the analytic feed-forward is
    # neutral-to-harmful, so it is off by default.  The reason is physical:
    # the lateral cascade already rejects a slowly-varying disturbance over a
    # 20 s descent, so an extra open-loop term mostly fights the feedback; and
    # the navigation EKF's disturbance state absorbs thrust-lag mismatch as
    # well as drag, so cancelling it injects a spurious term.  Prediction
    # therefore contributes through the *learned* policy, which receives the
    # forecast as observation features and can weight it as it sees fit.
    # Kept available for the ablation that demonstrates the above.
    use_disturbance_ff: bool = False
    dist_ff_gain: float = 1.0
    #: altitude above the pad at which the feed-forward starts to fade in,
    #: and the altitude at which it reaches full authority
    ff_schedule_alt: float = 900.0
    ff_full_alt: float = 150.0


# --------------------------------------------------------------------------- #
#  Environment / curriculum
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    dt_ctrl: float = 0.05  # guidance update period [s] (20 Hz)
    dt_phys: float = 0.01  # physics integration step [s] (100 Hz)
    coast_time: float = 0.0  # optional extra unpowered phase [s]

    # reward weights.  ``w_quality`` scales the logarithmic landing-quality
    # potential (see ``RecoveryEnv._potential``); the potential's gradient near
    # the pad must be O(1) per metre rather than O(1e-4) so the policy actually
    # receives a signal about the landing tolerance.
    #
    # The *magnitude* of the potential matters as much as its shape.  With
    # ``w_quality = 60`` the initial offset (up to 600 m) makes the telescoped
    # shaping term ``Phi(s_final) - Phi(s_0)`` alone ~+600, so the episode return
    # is O(600) and the +-100 terminal bonus is only ~15% of it.  A value
    # function cannot track returns of that size across the initial-condition
    # envelope: the shared trunk is dominated by the value regression and the
    # actor head, reading corrupted features, drifts into a *harmful* residual
    # (measured: 0% success vs 55% for the analytic baseline, critic error >500).
    # At 12 the return is O(200), the terminal bonus is ~40% of it, and the pad
    # gradient is still ~4/m.
    w_quality: float = 12.0
    w_pos: float = 50.0   # retained for the (unused) quadratic variant
    w_vel: float = 50.0
    w_fuel: float = 0.0   # fuel is reported, not penalised
    w_tilt: float = 0.05
    w_residual: float = 0.05
    w_tgo: float = 0.0
    r_success: float = 100.0
    r_crash: float = -100.0
    r_partial_pos: float = 0.0
    r_partial_vel: float = 0.0

    obs_clip: float = 10.0
    curriculum_stages: int = 4


@dataclass
class PPOConfig:
    total_steps: int = 1_200_000
    rollout_steps: int = 4096
    epochs: int = 6
    minibatch: int = 256
    #: The recovery task is episodic and its episodes are ~510 control steps
    #: long, so ``gamma = 0.995`` discounts the terminal +-100 bonus to
    #: ``0.995**510 ~= 0.08`` -- the one reward term that distinguishes a landing
    #: from a crash is effectively erased at the episode start.  A discount below
    #: 1 also breaks the policy-invariance of potential-based shaping: the
    #: telescoped term picks up a ``-(1-gamma) * sum(gamma^t * Phi_t)`` bias of
    #: order +270 here, which varies with the trajectory and rewards *lingering
    #: far from the pad*.  ``gamma = 1.0`` restores exact telescoping
    #: (``G = Phi_T - Phi_0``) and gives the terminal bonus its full weight.
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    lr_final_frac: float = 0.15
    entropy_coef: float = 4e-3
    value_coef: float = 0.6
    max_grad_norm: float = 0.6
    hidden: int = 160
    seed: int = 0
    device: str = "cpu"
    log_every: int = 5
    save_every: int = 25


# --------------------------------------------------------------------------- #
#  Top-level
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    rocket: RocketConfig = field(default_factory=RocketConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def air_density(altitude: np.ndarray | float) -> np.ndarray | float:
    """Exponential-atmosphere density model (valid to ~60 km)."""
    h = np.clip(altitude, 0.0, 60000.0)
    return RHO0 * np.exp(-h / SCALE_HEIGHT)
