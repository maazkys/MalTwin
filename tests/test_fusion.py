# tests/test_fusion.py
"""
Tests for hybrid fusion. The decision-table cases here are the same ones
listed in the Business Rules Testing section of the report.
"""
import pytest

from modules.fusion import fuse, merge_mitre_sources


class TestDecisionTable:
    # (static_conf, dynamic_score, expected_category, expected_verdict, escalated)
    @pytest.mark.parametrize('s,d,cat,verdict,esc', [
        (0.95, 0.90, 'CONFIRMED',     'MALICIOUS',  False),   # TC-FUS-1
        (0.92, 0.10, 'STATIC_ONLY',   'SUSPICIOUS', False),   # TC-FUS-2  fused 0.51
        (0.30, 0.95, 'NOVEL_VARIANT', 'SUSPICIOUS', False),   # TC-FUS-3  fused 0.625
        (0.10, 0.60, 'NOVEL_VARIANT', 'SUSPICIOUS', True),    # TC-FUS-4  fused 0.35 → escalated
        (0.20, 0.05, 'NO_EVIDENCE',   'BENIGN',     False),   # TC-FUS-5
        (0.85, 0.55, 'CONFIRMED',     'MALICIOUS',  False),   # TC-FUS-6  fused 0.70 boundary
    ])
    def test_cases(self, s, d, cat, verdict, esc):
        r = fuse(static_conf=s, dynamic_score=d)
        assert r.category == cat
        assert r.verdict == verdict
        assert r.escalated is esc
        assert r.fused_score == pytest.approx((s + d) / 2)


class TestSingleSource:
    def test_static_only_source(self):
        r = fuse(static_conf=0.9)
        assert r.category == 'STATIC_SOURCE_ONLY' and r.fused_score == pytest.approx(0.9)
        assert r.verdict == 'MALICIOUS'

    def test_dynamic_only_source(self):
        r = fuse(dynamic_score=0.2)
        assert r.category == 'DYNAMIC_SOURCE_ONLY' and r.verdict == 'BENIGN'


class TestValidation:
    def test_no_inputs(self):
        with pytest.raises(ValueError):
            fuse()

    @pytest.mark.parametrize('bad', [-0.1, 1.1])
    def test_out_of_range(self, bad):
        with pytest.raises(ValueError):
            fuse(static_conf=bad, dynamic_score=0.5)

    def test_negative_weight(self):
        with pytest.raises(ValueError):
            fuse(static_conf=0.5, dynamic_score=0.5, w_static=-1)

    def test_custom_weights(self):
        r = fuse(static_conf=1.0, dynamic_score=0.0, w_static=3, w_dynamic=1)
        assert r.fused_score == pytest.approx(0.75)


class TestMitreMerge:
    def test_corroborated_first_and_tagged(self):
        static = {'techniques': [{'id': 'T0846', 'name': 'Remote System Discovery'},
                                 {'id': 'T0859', 'name': 'Valid Accounts'}]}
        dynamic = {'techniques': [{'id': 'T0846', 'name': 'Remote System Discovery'},
                                  {'id': 'T0814', 'name': 'Denial of Service'}]}
        merged = merge_mitre_sources(static, dynamic)
        assert merged[0]['id'] == 'T0846'
        assert merged[0]['sources'] == ['static', 'dynamic']
        assert {m['id'] for m in merged} == {'T0846', 'T0859', 'T0814'}

    def test_handles_none(self):
        assert merge_mitre_sources(None, None) == []

    def test_result_serialisable(self):
        import json
        json.dumps(fuse(static_conf=0.5, dynamic_score=0.5, family='Allaple.A').to_dict())
