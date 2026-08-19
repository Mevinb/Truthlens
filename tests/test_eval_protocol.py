"""Guards on the evaluation protocol itself.

The protocol is the only thing standing between a plausible number and a
misleading one, so its two load-bearing behaviours are pinned here: a checkpoint
path that resolves however it was spelled, and tiers that cannot silently
include a source the model trained on.
"""

import json

import pytest


# ─── Checkpoint resolution ────────────────────────────────────────────────────
# Regression: --checkpoint was assigned straight to cfg.cnn_model_name, which
# every consumer joins onto models_dir. Passing the natural
# "models/resnet18_modern.pth" produced "models/models/resnet18_modern.pth" and
# diag_eval_all reported "nothing scored" — indistinguishable from an empty
# dataset. All three spellings must land on the same file.
@pytest.mark.parametrize("spelling", ["bare", "repo_relative", "absolute"])
def test_set_cnn_checkpoint_accepts_every_spelling(spelling):
    from src.utils import Config, set_cnn_checkpoint

    cfg = Config()
    available = sorted(cfg.models_dir.glob("*.pth"))
    if not available:
        pytest.skip("no checkpoint on disk to resolve")
    target = available[0]

    value = {
        "bare": target.name,
        "repo_relative": str(target.relative_to(cfg.models_dir.parent)),
        "absolute": str(target),
    }[spelling]

    set_cnn_checkpoint(cfg, value)
    assert (cfg.models_dir / cfg.cnn_model_name) == target


def test_set_cnn_checkpoint_raises_on_a_missing_file():
    from src.utils import Config, set_cnn_checkpoint

    with pytest.raises(FileNotFoundError, match="not found"):
        set_cnn_checkpoint(Config(), "definitely_not_a_checkpoint.pth")


# ─── Tier construction ────────────────────────────────────────────────────────
def _corpus(tmp_path, records):
    corpus = tmp_path / "corpus"
    for r in records:
        p = corpus / r["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    (corpus / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return corpus


def _rec(split, label, source_key, generator, name):
    return {"split": split, "label": label, "source_key": source_key,
            "generator": generator, "path": f"{split}/{label}/{name}.png"}


def test_a_trained_source_never_lands_in_a_held_out_tier(tmp_path, capsys):
    """A holdout row from a source the model trained on is a leak, not evidence.

    It is an easy one to create by accident — quarantining a repository at
    collection time is a flag on a source, and the same repository can be added
    twice under two keys. If it ever happens, the row must be excluded from the
    held-out tiers and the operator must be told, not quietly averaged in.
    """
    from src.eval_protocol import build_tiers

    corpus = _corpus(tmp_path, [
        _rec("train", "fake", "gen_a_wild", "gen-a", "t1"),
        _rec("train", "real", "cam_a", "camera/a", "t2"),
        _rec("holdout", "fake", "gen_a_wild", "gen-a", "leak"),      # trained source
        _rec("holdout", "fake", "gen_a_clean", "gen-a", "ok"),       # unseen source
        _rec("holdout", "real", "cam_b", "camera/b", "neg"),
    ])

    tiers = build_tiers(corpus=corpus, wild_corpus=tmp_path / "no_wild_set")

    held_out_rows = [r for t in tiers.values() if t.headline for r in t.rows]
    assert held_out_rows, "expected at least one held-out tier"
    assert all(r["source_key"] != "gen_a_wild" for r in held_out_rows)
    assert "leak" in capsys.readouterr().err


def test_an_unseen_generator_from_a_trained_repo_is_valid_t3(tmp_path, capsys):
    """Sharing a repository with training does not disqualify a new generator.

    A multi-generator repo where some generators are held out is the *cleanest*
    form of the unseen-generator question, not a contaminated one: packaging,
    compression and rendition are all held constant, so the generator is the
    only variable. Dropping these rows as a "leak" would discard the best
    evidence available and quietly shrink T3.
    """
    from src.eval_protocol import build_tiers

    corpus = _corpus(tmp_path, [
        _rec("train", "fake", "multi_gen_repo", "gen-a", "t1"),
        _rec("train", "real", "cam_a", "camera/a", "t2"),
        # Same repository as training, generator never trained on.
        _rec("holdout", "fake", "multi_gen_repo", "gen-z", "unseen_gen"),
        _rec("holdout", "real", "cam_b", "camera/b", "neg"),
    ])

    tiers = build_tiers(corpus=corpus, wild_corpus=tmp_path / "no_wild_set")

    assert "T3_xgen" in tiers, "an unseen generator must still form a tier"
    assert {r["generator"] for r in tiers["T3_xgen"].rows if r["label"] == "fake"} \
        == {"gen-z"}
    # And it must not be reported as a leak, because nothing leaked.
    assert "leak" not in capsys.readouterr().err
    # The tier says out loud that provenance is shared, so the number is not
    # mistaken for one that also varies the source.
    assert "multi_gen_repo" in tiers["T3_xgen"].why


def test_tiers_separate_unseen_source_from_unseen_generator(tmp_path):
    """T2 and T3 answer different questions and must not be pooled.

    The shipped model scores 53.5% on T2 and 95.5% on T3. Pooling them reports
    87.1% and hides the only failure that matters.
    """
    from src.eval_protocol import build_tiers

    corpus = _corpus(tmp_path, [
        _rec("train", "fake", "gen_a_wild", "gen-a", "t1"),
        _rec("train", "real", "cam_a", "camera/a", "t2"),
        _rec("test", "fake", "gen_a_wild", "gen-a", "e1"),
        _rec("test", "real", "cam_a", "camera/a", "e2"),
        _rec("holdout", "fake", "gen_a_clean", "gen-a", "xsource"),  # trained gen
        _rec("holdout", "fake", "gen_b", "gen-b", "xgen"),           # unseen gen
        _rec("holdout", "real", "cam_b", "camera/b", "neg"),
    ])

    tiers = build_tiers(corpus=corpus, wild_corpus=tmp_path / "no_wild_set")
    assert set(tiers) == {"T1_indist", "T2_xsource", "T3_xgen"}
    assert not tiers["T1_indist"].headline, "in-distribution is not evidence"

    def fakes(key):
        return {r["source_key"] for r in tiers[key].rows if r["label"] == "fake"}

    assert fakes("T2_xsource") == {"gen_a_clean"}
    assert fakes("T3_xgen") == {"gen_b"}
    # Both borrow the same unseen-source reals, and say so.
    for key in ("T2_xsource", "T3_xgen"):
        assert tiers[key].n_real == 1
        assert "cam_b" in tiers[key].negatives


# ─── Metrics ──────────────────────────────────────────────────────────────────
def test_threshold_at_fpr_holds_the_false_alarm_budget():
    import numpy as np

    from src.eval_protocol import at_threshold, threshold_at_fpr

    # 20 reals low, 20 fakes high, with four fakes buried inside the real range.
    y = np.array([0] * 20 + [1] * 20)
    p = np.concatenate([np.linspace(0.01, 0.40, 20),
                        np.concatenate([np.linspace(0.05, 0.35, 4),
                                        np.linspace(0.60, 0.99, 16)])])

    thr = threshold_at_fpr(y, p, 0.05)
    scored = at_threshold(y, p, thr)
    assert scored["fpr"] <= 0.05 + 1e-9
    assert scored["fake_recall"] == pytest.approx(0.8, abs=0.05)


def test_recall_at_fpr_is_unreachable_when_reals_outrank_every_fake():
    """Reported as unreachable rather than as a silently degraded threshold."""
    import numpy as np

    from src.eval_protocol import threshold_at_fpr

    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
    assert not np.isfinite(threshold_at_fpr(y, p, 0.05))
