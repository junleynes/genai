from app import generation as g

DEFAULTS = {"model_type": "ltx2_22B_distilled_1_1", "guidance_scale": 7.5, "resolution": "1280x704"}


def _job(**params):
    base = {"resolution": "1920x1080", "duration_seconds": 10, "guidance_scale": None}
    base.update(params)
    return {"id": "t", "job_type": "i2v", "prompt": "rows slide in", "image_path": "/x.png", "params": base}


def test_preserve_layout_pins_end_frame_and_steers_prompt():
    s = g._map_job_to_settings(_job(preserve_layout=True), DEFAULTS)
    assert s["image_end"] == "/x.png" and "E" in s["image_prompt_type"]
    assert g.PRESERVE_LAYOUT_SUFFIX in s["prompt"] and s["prompt"].startswith("rows slide in.")
    assert s["negative_prompt"] == g.PRESERVE_LAYOUT_NEGATIVE
    # idempotent, and a user-supplied negative is kept
    assert g._preserve_layout_prompt(s["prompt"]) == s["prompt"]
    s2 = g._map_job_to_settings(_job(preserve_layout=True, negative_prompt="mine"), DEFAULTS)
    assert s2["negative_prompt"] == "mine"


def test_preserve_layout_off_changes_nothing():
    s = g._map_job_to_settings(_job(), DEFAULTS)
    assert "image_end" not in s and s["prompt"] == "rows slide in"


def test_explicit_end_image_beats_preserve_default():
    j = _job(preserve_layout=True, end_image_url="/end.png")
    s = g._map_job_to_settings(j, DEFAULTS)
    assert s["image_end"] == "/end.png"


def test_loop_plan_and_seam_flag():
    j = _job(preserve_layout=True, loop_to_seconds=15, duration_seconds=5)
    s = g._map_job_to_settings(j, DEFAULTS)
    post, notes = g._finalize_sampling(s, j, {})
    assert post["loop_to"] == 15 and post["drop_last_frame"] is True
    assert post["crop_to"] == [1920, 1080] and s["resolution"] == "1920x1088"
    # without preserve the seam warning is shown and no frame is dropped
    j2 = _job(loop_to_seconds=15, duration_seconds=5)
    s2 = g._map_job_to_settings(j2, DEFAULTS)
    post2, notes2 = g._finalize_sampling(s2, j2, {})
    assert post2["loop_to"] == 15 and "drop_last_frame" not in post2
    assert any("seam" in n for n in notes2)


def test_loop_not_longer_than_clip_is_ignored_with_note():
    j = _job(loop_to_seconds=4, duration_seconds=5)
    s = g._map_job_to_settings(j, DEFAULTS)
    post, notes = g._finalize_sampling(s, j, {})
    assert "loop_to" not in post and any("nothing was looped" in n for n in notes)


def test_blank_seed_does_not_crash():
    assert g._map_job_to_settings(_job(seed=""), DEFAULTS)["seed"] == -1


def test_fit_pipeline_replaces_start_and_end_with_fitted_file(tmp_path, monkeypatch):
    from PIL import Image
    src = tmp_path / "in.png"
    Image.new("RGB", (1916, 1018), (0, 100, 200)).save(src)
    monkeypatch.setattr(g, "UPLOAD_DIR", tmp_path / "up")
    j = _job(preserve_layout=True)
    j["image_path"] = str(src)
    s = g._map_job_to_settings(j, DEFAULTS)
    post, notes = g._finalize_sampling(s, j, {})
    g._fit_source_images(s, j, {}, post, notes)
    assert s["image_start"] == s["image_end"] != str(src)
    assert Image.open(s["image_start"]).size == (1920, 1088)
    assert any("Start image was 1916x1018" in n for n in notes)
