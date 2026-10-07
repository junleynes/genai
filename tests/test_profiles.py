from app import profiles as P


def test_family_detection():
    assert P.family_of("ltx2_22B_distilled_1_1") == "ltx2_distilled"
    assert P.family_of("ltx2_22B") == "ltx2"
    assert P.family_of("t2v_2_2") == "default"
    assert P.family_of("vace_14B") == "wan"
    assert P.family_of("") == "default"
    assert P.family_of(None) == "default"


def test_cfg_explicit_zero_is_honoured():
    assert P.resolve_cfg("ltx2_22B_distilled_1_1", 0) == 0.0
    assert P.resolve_cfg("flux_dev", 0, admin_default=7.5) == 0.0


def test_cfg_auto_uses_profile_then_admin_then_fallback():
    assert P.resolve_cfg("ltx2_22B_distilled_1_1", None, 7.5) == 1.0
    assert P.resolve_cfg("ltx2_22B_distilled_1_1", "", 7.5) == 1.0
    assert P.resolve_cfg("vace_14B", None, 7.5) == 7.5
    assert P.resolve_cfg("vace_14B", None, None) == 5.0
    assert P.resolve_cfg("vace_14B", "garbage", 6) == 6.0


def test_negative_prompt_effectiveness():
    assert not P.negative_is_effective("ltx2_22B_distilled_1_1", 1.0)
    assert not P.negative_is_effective("ltx2_22B_distilled_1_1", 4.0)  # distilled never uses it
    assert P.negative_is_effective("ltx2_22B", 3.0)
    assert not P.negative_is_effective("ltx2_22B", 1.0)
    assert P.negative_is_effective("vace_14B", 5.0)


def test_snap_resolution_rounds_up_and_reports_request():
    res, req = P.snap_resolution("1920x1080", "ltx2_22B_distilled_1_1")
    assert res == "1920x1088" and req == (1920, 1080)
    res, req = P.snap_resolution("1280x704", "ltx2_22B_distilled_1_1")
    assert res == "1280x704" and req is None
    # families with no grid requirement are left alone
    assert P.snap_resolution("1920x1080", "vace_14B") == ("1920x1080", None)
    # unparsable input is passed through untouched
    assert P.snap_resolution("auto", "ltx2_22B") == ("auto", None)


def test_snap_frames_is_valid_for_family():
    for n in (49, 61, 97, 120, 241, 361):
        k = P.snap_frames(n, "ltx2_22B_distilled_1_1")
        assert (k - 1) % 8 == 0 and abs(k - n) <= 4
    assert P.snap_frames(241, "ltx2_22B_distilled_1_1") == 241
    assert (P.snap_frames(121, "vace_14B") - 1) % 4 == 0
    assert P.snap_frames(1, "ltx2_22B") == 9
