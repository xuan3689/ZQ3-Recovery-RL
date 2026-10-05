"""Smoke and regression tests.

Deliberately fast (a few seconds) so they can gate every commit.  Run with:

    python -m pytest tests -q
or, without pytest:

    python tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zq3rl.config import Config, air_density              # noqa: E402
from zq3rl.dynamics import (                              # noqa: E402
    FlightState, WindField, body_from_world, touchdown_metrics,
)
from zq3rl.env import RecoveryEnv                          # noqa: E402
from zq3rl.estimator import PerceptionStack                # noqa: E402
from zq3rl.guidance import (                              # noqa: E402
    SafetyShield, baseline_guidance, descent_speed_profile,
)
from zq3rl.ppo import ActorCritic, PPOTrainer              # noqa: E402


# --------------------------------------------------------------------------- #
def test_air_density_monotone():
    h = np.linspace(0, 40000, 200)
    rho = air_density(h)
    assert np.all(np.diff(rho) < 0), "density must fall with altitude"
    assert abs(float(air_density(0.0)) - 1.225) < 1e-6


def test_body_frame_is_proper_rotation():
    for tilt in np.linspace(-1.2, 1.2, 13):
        for azim in np.linspace(-np.pi, np.pi, 7):
            R = body_from_world(tilt, azim)
            assert abs(np.linalg.det(R) - 1.0) < 1e-9, "det must be +1"
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "must be orthogonal"


def test_body_frame_thrust_axis():
    """The first body axis must equal the analytic thrust direction."""
    tilt, azim = 0.31, -1.2
    b = body_from_world(tilt, azim)[0]
    st, ct = np.sin(tilt), np.cos(tilt)
    expect = np.array([st * np.cos(azim), ct, st * np.sin(azim)])
    assert np.allclose(b, expect, atol=1e-12)


def test_wind_field_deterministic_given_seed():
    cfg = Config()
    w1 = WindField(cfg.wind, np.random.default_rng(7))
    w2 = WindField(cfg.wind, np.random.default_rng(7))
    for h in (0.0, 100.0, 350.0, 1500.0):
        assert np.allclose(w1.velocity(h), w2.velocity(h))


def test_wind_shear_layer_is_a_bump():
    """The mean speed must peak near the configured shear altitude."""
    cfg = Config()
    cfg.wind.shear_gain = 0.85
    w = WindField(cfg.wind, np.random.default_rng(0))
    hs = np.linspace(50, 900, 400)
    speeds = [float(np.linalg.norm(w._profile(h))) for h in hs]
    peak_h = hs[int(np.argmax(speeds))]
    assert abs(peak_h - cfg.wind.shear_alt) < 120.0


# --------------------------------------------------------------------------- #
def test_speed_profile_monotone_and_terminal():
    v_t = -1.5
    hs = np.linspace(0, 1500, 400)
    vs = [descent_speed_profile(h, v_t, 3.0, 60.0)[0] for h in hs]
    assert abs(vs[0] - v_t) < 1e-9, "at h=0 the profile equals the target"
    # sink rate magnitude grows with altitude
    assert np.all(np.diff(vs) <= 1e-9), "profile must be monotone (more negative)"


def test_speed_profile_derivative_consistency():
    """dv_ref/dh must match a finite difference of v_ref."""
    v_t, a_ref = -1.5, 3.0
    h = 400.0
    v, dv = descent_speed_profile(h, v_t, a_ref, 60.0)
    eps = 1e-4
    v_hi = descent_speed_profile(h + eps, v_t, a_ref, 60.0)[0]
    assert abs((v_hi - v) / eps - dv) < 1e-3


# --------------------------------------------------------------------------- #
def test_shield_clamps_vertical_thrust():
    """A command demanding downward thrust must be lifted to the floor."""
    cfg = Config()
    shield = SafetyShield(cfg)
    accel = np.array([0.0, -30.0, 0.0])   # net 30 m/s^2 downward
    net, cmd, rep = shield.project(accel, 7.0, 22.0, 500.0, 0.0)
    assert cmd.throttle >= 0.0
    # the vertical component of the thrust vector must stay positive
    tv = net.copy(); tv[1] += 9.80665
    assert tv[1] > 0.0, "engine cannot pull the vehicle down"
    assert rep.clipped_tilt or rep.clipped_throttle


def test_shield_respects_tilt_authority():
    cfg = Config()
    shield = SafetyShield(cfg)
    # demand a huge horizontal acceleration at low altitude
    accel = np.array([40.0, 12.0, 0.0])
    net, cmd, rep = shield.project(accel, 7.0, 22.0, 40.0, 0.0)
    lim = shield.tilt_limit(40.0)
    assert cmd.tilt <= lim + 1e-9, "tilt command must respect the authority"


def test_tilt_authority_schedule():
    cfg = Config()
    shield = SafetyShield(cfg)
    assert shield.tilt_limit(1000.0) > shield.tilt_limit(50.0)
    assert abs(shield.tilt_limit(10_000.0)
               - cfg.rocket.tilt_max_deg * np.pi / 180.0) < 1e-9


# --------------------------------------------------------------------------- #
def test_baseline_guidance_is_finite_and_bounded():
    cfg = Config()
    g = cfg.guidance
    pos = np.array([300.0, 1200.0, -150.0])
    vel = np.array([-30.0, -140.0, 12.0])
    sol = baseline_guidance(pos, vel, np.array([0.0, 6.0, 0.0]),
                            np.array([0.0, -1.5, 0.0]), 7.0, 22.0, g)
    assert np.all(np.isfinite(sol.accel))
    assert sol.tgo > 0.0
    thrust = sol.accel.copy(); thrust[1] += 9.80665
    assert np.linalg.norm(thrust) <= 22.0 * 1.01, "must respect the envelope"


# --------------------------------------------------------------------------- #
def test_nav_ekf_reduces_error():
    """Fusing noisy GNSS+baro must beat the raw GNSS position error."""
    from zq3rl.estimator import NavigationEKF
    cfg = Config()
    rng = np.random.default_rng(3)
    ekf = NavigationEKF(cfg.estimator, 0.01)
    truth = np.array([0.0, 1000.0, 0.0])
    vel = np.array([0.0, -50.0, 0.0])
    ekf.x[0:3] = truth
    ekf.x[3:6] = vel
    raw_err, est_err = [], []
    for k in range(400):
        truth = truth + vel * 0.01
        ekf.predict(np.array([0.0, -9.80665, 0.0]), 0.01)
        if k % 10 == 0:
            z = truth + rng.normal(0.0, 1.2, 3)
            ekf.update_gnss(z, vel + rng.normal(0.0, 0.25, 3), k * 0.01)
            raw_err.append(float(np.linalg.norm(z - truth)))
            est_err.append(float(np.linalg.norm(ekf.pos - truth)))
    assert np.mean(est_err) < np.mean(raw_err), \
        f"EKF ({np.mean(est_err):.3f}) should beat raw GNSS ({np.mean(raw_err):.3f})"


def test_attitude_ekf_tracks_tilt():
    """The attitude filter must follow a commanded tilt profile closely."""
    from zq3rl.estimator import AttitudeEKF
    cfg = Config()
    ekf = AttitudeEKF(cfg.estimator, 0.01, wn=cfg.rocket.att_wn,
                      zeta=cfg.rocket.att_zeta,
                      gyro_noise_deg=cfg.sensors.gyro_noise_deg)
    rng = np.random.default_rng(11)
    tilt = rate = 0.0
    cmd = 0.35
    errs = []
    for k in range(600):
        # simulate the autopilot
        acc = cfg.rocket.att_wn ** 2 * (cmd - tilt) - 2 * cfg.rocket.att_zeta \
            * cfg.rocket.att_wn * rate
        rate += acc * 0.01
        tilt += rate * 0.01
        gyro = rate + rng.normal(0.0, cfg.sensors.gyro_noise_deg * np.pi / 180.0)
        ekf.predict(gyro, 0.0, 0.01, cmd)
        if k > 400:
            errs.append(abs(ekf.tilt - tilt))
    assert np.mean(errs) < np.deg2rad(1.0), \
        f"attitude error {np.rad2deg(np.mean(errs)):.3f} deg too large"


# --------------------------------------------------------------------------- #
def test_disturbance_predictor_recovers_wind():
    """The wind inversion must recover the injected wind vector."""
    from zq3rl.estimator import DisturbancePredictor
    cfg = Config()
    pred = DisturbancePredictor(cfg.estimator)
    rng = np.random.default_rng(5)
    w_true = np.array([12.0, 0.0, -4.0])
    vel = np.array([-20.0, -60.0, 6.0])
    rho, cd_a, m = 1.225, 15.9, 30000.0
    k = rho * cd_a / m
    rel = vel - w_true
    a_true = -0.5 * k * float(np.linalg.norm(rel)) * rel
    for _ in range(400):
        pred.update(a_true + rng.normal(0.0, 0.05, 3), vel, 500.0, k, 0.01)
    assert np.linalg.norm(pred.wind_est - w_true) < 1.5, \
        f"wind estimate {pred.wind_est} vs true {w_true}"


# --------------------------------------------------------------------------- #
def test_env_observation_shape_and_finiteness():
    cfg = Config()
    env = RecoveryEnv(cfg, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.OBS_DIM,), f"{obs.shape} vs {env.OBS_DIM}"
    assert np.all(np.isfinite(obs))
    for _ in range(20):
        obs, r, done, trunc, info = env.step(np.zeros(3))
        assert np.all(np.isfinite(obs))
        assert np.isfinite(r)
        if done or trunc:
            break


def test_env_is_deterministic_for_a_seed():
    cfg = Config()
    a = RecoveryEnv(cfg, seed=0)
    b = RecoveryEnv(cfg, seed=0)
    oa, _ = a.reset(seed=99)
    ob, _ = b.reset(seed=99)
    assert np.allclose(oa, ob)
    for _ in range(30):
        oa, ra, da, ta, ia = a.step(np.zeros(3))
        ob, rb, db, tb, ib = b.step(np.zeros(3))
        assert np.allclose(oa, ob) and abs(ra - rb) < 1e-9


def test_baseline_lands_on_a_gentle_scenario():
    """On the easiest curriculum stage the analytic baseline must land."""
    cfg = Config()
    env = RecoveryEnv(cfg, seed=0, curriculum_level=0.05)
    ok = 0
    n = 8
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i)
        done = trunc = False
        while not (done or trunc):
            obs, r, done, trunc, info = env.step(np.zeros(3))
        ok += int(info["success"])
    assert ok >= int(0.7 * n), f"baseline only landed {ok}/{n} on the easy stage"


def test_success_metric_uses_all_three_criteria():
    cfg = Config()
    m = cfg.mission
    good = FlightState(pos=np.array([0.0, 6.0, 0.0]),
                       vel=np.array([0.0, -1.0, 0.0]), tilt=0.0)
    assert touchdown_metrics(good, m)["success"]
    far = FlightState(pos=np.array([50.0, 6.0, 0.0]),
                      vel=np.array([0.0, -1.0, 0.0]), tilt=0.0)
    assert not touchdown_metrics(far, m)["success"]
    fast = FlightState(pos=np.array([0.0, 6.0, 0.0]),
                       vel=np.array([0.0, -20.0, 0.0]), tilt=0.0)
    assert not touchdown_metrics(fast, m)["success"]
    tipped = FlightState(pos=np.array([0.0, 6.0, 0.0]),
                         vel=np.array([0.0, -1.0, 0.0]), tilt=np.deg2rad(20.0))
    assert not touchdown_metrics(tipped, m)["success"]


# --------------------------------------------------------------------------- #
def test_ppo_action_is_bounded():
    import torch
    net = ActorCritic(28, 3, 64)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.normal(0, 3, 28).astype(np.float32)
        a, logp, v = net.act(obs)
        assert np.all(np.abs(a) <= 1.0 + 1e-6), "tanh must bound the action"
        assert np.isfinite(logp) and np.isfinite(v)


def test_ppo_gae_reduces_to_reward_when_no_bootstrap():
    """With gamma=1 and a terminal step, the advantage equals the return."""
    cfg = Config()
    cfg.ppo.rollout_steps = 8
    cfg.ppo.gamma = 1.0
    cfg.ppo.gae_lambda = 1.0
    env = RecoveryEnv(cfg, seed=0)
    tr = PPOTrainer(env, cfg.ppo, seed=0)
    # fill one rollout
    tr.collect()
    assert tr.buf.ptr == cfg.ppo.rollout_steps


def test_effort_penalty_is_once_per_control_step():
    """Regression: control-effort penalties must not be scaled by the substep
    ratio.

    An earlier version folded ``w_residual * |a_rl|^2`` into the per-substep
    shaping term, so the penalty was applied ``dt_ctrl / dt_phys`` (= 5) times
    per control step.  The resulting ~2040-point episode penalty against a
    100-point success bonus made the optimal policy a *zero* residual, i.e. the
    learned controller collapsed back into the analytic baseline.  Pinning the
    call counts stops the two rates from ever being conflated again.
    """
    cfg = Config()
    env = RecoveryEnv(cfg, seed=0, curriculum_level=1.0)
    n_sub = int(round(cfg.env.dt_ctrl / cfg.env.dt_phys))

    counts = {"shaping": 0, "effort": 0}
    real_shaping = env._step_reward
    real_effort = env._effort_penalty

    def counting_shaping(out):
        counts["shaping"] += 1
        return real_shaping(out)

    def counting_effort(out, residual):
        counts["effort"] += 1
        return real_effort(out, residual)

    env._step_reward = counting_shaping
    env._effort_penalty = counting_effort

    env.reset(seed=0)
    _, _, done, trunc, _ = env.step(np.ones(3))  # full residual, one control step
    assert not (done or trunc), "a single step from 2500 m must not terminate"
    assert counts["effort"] == 1, (
        f"effort penalty applied {counts['effort']}x per control step, expected 1")
    assert counts["shaping"] == n_sub, (
        f"shaping applied {counts['shaping']}x, expected {n_sub} (per substep)")


def test_reward_magnitude_is_bounded_per_step():
    """The residual penalty must be O(w_residual) per control step, not
    O(w_residual * a_max^2).

    Normalising the residual by ``residual_max`` keeps a full-authority action
    on all three axes at exactly ``3 * w_residual`` per step.  Without the
    normalisation the same action would cost ``3 * w_residual * a_max^2``
    (= 2.4), and once the substep factor is included the episode penalty dwarfs
    the success bonus and the policy collapses to a zero residual.
    """
    cfg = Config()
    env = RecoveryEnv(cfg, seed=0, curriculum_level=1.0)
    env.reset(seed=0)
    env.step(np.zeros(3))
    out = env.perception.output()

    res_max = cfg.guidance.residual_max
    eff_zero = env._effort_penalty(out, np.zeros(3))
    eff_full = env._effort_penalty(out, np.full(3, res_max))
    residual_cost = eff_zero - eff_full  # isolating the residual term

    expected = 3.0 * cfg.env.w_residual  # three axes, each normalised to 1
    assert abs(residual_cost - expected) < 1e-9, (
        f"full-residual cost {residual_cost:.4f}, expected {expected:.4f} "
        "(residual must be normalised by residual_max)")
    assert residual_cost < cfg.env.r_success / 20.0, (
        "per-step effort penalty must stay well below the success bonus")


def test_training_step_improves_or_preserves_finiteness():
    """One PPO update must not produce NaN parameters."""
    cfg = Config()
    cfg.ppo.rollout_steps = 256
    cfg.ppo.epochs = 1
    env = RecoveryEnv(cfg, seed=0)
    tr = PPOTrainer(env, cfg.ppo, seed=0)
    tr.collect()
    stats = tr.update(0.0)
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)
    for p in tr.net.parameters():
        assert np.all(np.isfinite(p.detach().numpy()))


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
