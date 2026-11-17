"""climb_compare: dynamic time warping, size-independent features, hold sequences and the
wall registration, on synthetic data."""

import cv2
import numpy as np
import pytest

import climb_compare as cc


# ---------------------------------------------------------------- DTW
def brute_dtw(C):
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[n, m]


def test_dtw_matches_the_textbook_recurrence():
    rng = np.random.default_rng(0)
    for n, m in [(1, 1), (1, 5), (5, 1), (7, 9), (20, 13)]:
        C = rng.random((n, m))
        path, total = cc.dtw(C)
        assert total == pytest.approx(brute_dtw(C))
        assert path[0] == (0, 0) and path[-1] == (n - 1, m - 1)
        steps = np.diff(np.array(path), axis=0)
        assert ((steps >= 0) & (steps <= 1)).all() and (steps.sum(1) >= 1).all()
        assert sum(C[i, j] for i, j in path) == pytest.approx(total)


def test_dtw_identical_sequences_is_the_diagonal():
    x = np.sin(np.linspace(0, 6, 50))[:, None]
    path, total = cc.dtw(cc.cost_matrix(x, x))
    assert total == pytest.approx(0, abs=1e-9)
    assert path == [(i, i) for i in range(50)]


def test_dtw_recovers_a_known_time_warp():
    """B is A played at a varying speed (slow, then fast) and cut short at both ends; the
    alignment must find, for each A sample, the B sample that shows it."""
    t_a = np.linspace(0, 1, 300)
    curve = lambda t: np.stack([t, np.sin(7 * t), np.cos(13 * t) * t], 1)   # noqa: E731
    A = curve(t_a)
    # B's sample j shows A-time g(j): slow start, fast middle, a pause
    u = np.linspace(0, 1, 420)
    g = 0.1 + 0.8 * (u + 0.12 * np.sin(2 * np.pi * u))
    B = curve(g)
    lo, hi = np.searchsorted(t_a, g[0]), np.searchsorted(t_a, g[-1])
    path, _ = cc.dtw(cc.cost_matrix(A[lo:hi], B))
    jm = cc.path_map(path, hi - lo)
    truth = np.interp(t_a[lo:hi], g, np.arange(len(g)))
    err = np.abs(jm - truth)
    assert np.median(err) <= 1.0 and err.max() <= 4


def test_dtw_rejects_nan():
    with pytest.raises(ValueError):
        cc.dtw(np.array([[0.0, np.nan]]))


# ---------------------------------------------------------------- features
def skeleton(n=40, scale=1.0, shift=(0, 0), seed=0):
    """A climber reaching upward: (n, 16, 2) wall points + visibility. Joint layout follows
    climb_pose.JOINTS; scale and shift make a bigger/smaller climber elsewhere."""
    rng = np.random.default_rng(seed)
    base = {11: (-20, -60), 12: (20, -60), 13: (-40, -90), 14: (40, -30), 15: (-45, -120), 16: (50, 0),
            19: (-46, -130), 20: (52, 8), 23: (-15, 0), 24: (15, 0), 25: (-30, 40), 26: (30, 45),
            27: (-35, 85), 28: (35, 90), 29: (-40, 92), 30: (40, 97), 31: (-30, 95), 32: (30, 100)}
    pts = np.array([base[i] for i in cc.JOINT_IDS], float)
    wall = []
    for f in range(n):
        p = pts.copy()
        p[cc.J[15]] += (0, -f)                 # left hand creeps up
        p[cc.J[19]] += (0, -f)
        p[:, 1] -= 3 * f                       # whole body climbs
        wall.append(p * scale + (500, 800) + shift)
    wall = np.array(wall) + rng.normal(0, 0.01, (n, len(cc.JOINT_IDS), 2))
    return wall, np.ones((n, len(cc.JOINT_IDS)))


def test_posture_features_ignore_body_size_and_position():
    wa, va = skeleton()
    wb, vb = skeleton(scale=1.4, shift=(37, -20))
    box = (300, 300, 600)
    na, nb = cc.normalise(wa, va, box), cc.normalise(wb, vb, box)
    assert nb["torso"] == pytest.approx(1.4 * na["torso"], rel=1e-3)
    np.testing.assert_allclose(nb["limbs"], na["limbs"], atol=1e-3)   # torso lengths
    np.testing.assert_allclose(nb["angles"], na["angles"], atol=0.05)  # degrees
    # hip position is on the route, so it does move with where the climber is
    assert not np.allclose(nb["hip"], na["hip"], atol=1e-3)


def test_angles_measured_on_the_wall():
    a, b, c = np.array([[0, 0]]), np.array([[1, 0]]), np.array([[1, 1]])
    assert cc.angle_deg(a, b, c)[0] == pytest.approx(90)
    assert np.isnan(cc.angle_deg(b, b, c)[0])


def test_hip_position_is_relative_to_the_route():
    wall, vis = skeleton(n=5)
    hip = (wall[:, cc.J[23]] + wall[:, cc.J[24]]) / 2
    x0, y0, h = 100.0, 200.0, 400.0
    nf = cc.normalise(wall, vis, (x0, y0, h))
    np.testing.assert_allclose(nf["hip"][2], (hip[2] - (x0, y0)) / h, atol=0.02)


def test_missing_and_unsure_joints_are_filled_from_neighbours():
    wall, vis = skeleton(n=30)
    wall[10:14] = np.nan                       # pose lost for 4 frames
    vis[20, cc.J[19]] = 0.1                    # one unsure hand
    nf = cc.normalise(wall, vis, (0, 0, 500))
    assert np.isfinite(nf["hip"]).all() and np.isfinite(nf["limbs"]).all() and np.isfinite(nf["angles"]).all()
    F = cc.feature_matrix(nf)
    assert F.shape == (30, 2 + 8 + 6) and np.isfinite(F).all()


def test_route_box_from_holds_or_fallback():
    holds = [{"id": 1, "center": [100, 900]}, {"id": 2, "center": [300, 100]}]
    assert cc.route_box(holds) == (100.0, 100.0, 800.0)
    assert cc.route_box([], np.array([[0, 0], [50, 200], [np.nan, np.nan]])) == (0.0, 0.0, 200.0)


# ---------------------------------------------------------------- holds per limb, sequences
def test_runs_drop_short_contacts_and_merge_repeats():
    seq = [{"l_hand": h} for h in [1] * 6 + [2] * 2 + [1] * 5 + [""] * 3 + [3] * 4]
    assert cc.runs(seq, "l_hand", 3) == [(1, 0, 13), (3, 16, 20)]


def test_sequence_differences_in_words():
    d = cc.sequence_diffs([1, 2, 4, 5], [1, 3, 4, 5, "V1"])
    assert d == ["A used hold 2 where B used hold 3", "only B used volume V1"]
    assert cc.sequence_diffs([1, 2], [1, 2]) == []


def test_hold_cost_only_when_both_on_different_holds():
    FA, FB = np.zeros((2, 1)), np.zeros((2, 1))
    ca = cc.hold_codes([{"l_hand": 1}, {"l_hand": ""}], [1, 2])
    cb = cc.hold_codes([{"l_hand": 2}, {"l_hand": 1}], [1, 2])
    C = cc.alignment_cost(FA, FB, ca, cb)
    assert C[0, 0] == pytest.approx(cc.W_HOLD) and C[0, 1] == 0 and C[1, 0] == 0


def test_climb_window_starts_at_the_first_hand_on_a_hold():
    lh = [{"l_hand": "", "r_hand": ""}] * 60 + [{"l_hand": 1, "r_hand": ""}] * 30 + [{"l_hand": "", "r_hand": ""}] * 60
    s, e = cc.climb_window(lh, np.ones(150), fps=30)
    assert s == 60 - 9 and e == 90 + 9


# ---------------------------------------------------------------- registration
def texture(seed, h=960, w=540):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((h // 4, w // 4)) * 255).astype(np.uint8), (3, 3), 0)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    for _ in range(150):                                  # "holds": blobs with corners
        x, y, r = rng.integers(0, w), rng.integers(0, h), rng.integers(4, 14)
        cv2.circle(img, (int(x), int(y)), int(r), int(rng.integers(0, 255)), -1)
    return img


def test_registration_recovers_a_known_camera_change():
    wall = texture(1)
    G = np.array([[0.92, 0.06, 30], [-0.04, 0.95, 50], [2e-5, 1e-5, 1]])   # B's view of A's wall
    b_img = cv2.warpPerspective(wall, G, (wall.shape[1], wall.shape[0]))
    full = np.ones_like(wall) * 255
    # A's base = full-res pixels at 2x; samples are half-size with to_base = scale 2
    S = np.diag([2.0, 2, 1])
    a_samples = [(wall, S, full), (wall, S, full)]
    b_samples = [(b_img, S, full), (b_img, S, full)]
    R, stats = cc.register_samples(a_samples, b_samples, frame_w=1080)
    truth = S @ np.linalg.inv(G) @ np.linalg.inv(S)       # B base -> A base
    pts = np.array([[200, 300], [800, 1500], [500, 900]], float)
    np.testing.assert_allclose(cc.warp_pts(pts, R), cc.warp_pts(pts, truth), atol=3)
    assert stats["frame_pairs_agreeing"] == 4


def test_registration_refuses_different_walls():
    full = np.ones((960, 540), np.uint8) * 255
    S = np.diag([2.0, 2, 1])
    with pytest.raises(cc.RegistrationError):
        cc.register_samples([(texture(1), S, full)] * 2, [(texture(2), S, full)] * 2)


def test_summary_is_plain_json():
    import json
    out = cc.plain({"a": np.float64(1.5), "b": [np.bool_(True), np.int64(3)], "c": np.zeros(2)})
    assert json.loads(json.dumps(out)) == {"a": 1.5, "b": [True, 3], "c": [0.0, 0.0]}
