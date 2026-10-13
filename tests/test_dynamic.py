# tests/test_dynamic.py
"""Tests for the dynamic (behavioural) analysis module."""
import io

import numpy as np
import pandas as pd
import pytest

from modules.dynamic import (
    parse_conn_log, load_conn_logs, map_behaviour, extract_flow_features,
    feature_names, DynamicDetector,
)
from modules.dynamic.synthetic import generate_flows
from modules.dynamic.zeek_parser import write_conn_log
from modules.dynamic.detector import load_behaviour_mitre, merge_mitre
import config


@pytest.fixture(scope='module')
def flows():
    df = generate_flows({'Benign': 400, 'PortScan': 120, 'C&C': 60, 'DDoS': 60,
                         'Attack': 50, 'FileDownload': 30, 'Okiru': 50}, seed=3)
    df['behaviour'] = [map_behaviour(a, b) for a, b in zip(df['label'], df['detailed-label'])]
    return df


@pytest.fixture(scope='module')
def trained(flows):
    return DynamicDetector(n_estimators=40, n_jobs=1).fit(flows)


class TestBehaviourMapping:
    @pytest.mark.parametrize('label,detail,expected', [
        ('benign', '-', 'Benign'),
        ('Benign', '-', 'Benign'),
        ('Malicious', 'PartOfAHorizontalPortScan', 'PortScan'),
        ('Malicious', 'PartOfAHorizontalPortScan-Attack', 'PortScan'),
        ('Malicious', 'C&C', 'C&C'),
        ('Malicious', 'C&C-HeartBeat', 'C&C'),
        ('Malicious', 'C&C-FileDownload', 'FileDownload'),
        ('Malicious', 'C&C-HeartBeat-FileDownload', 'FileDownload'),
        ('Malicious', 'DDoS', 'DDoS'),
        ('Malicious', 'Okiru', 'Okiru'),
        ('Malicious', 'Okiru-Attack', 'Okiru'),
        ('Malicious', 'Attack', 'Attack'),
        ('Malicious', '-', 'Attack'),
    ])
    def test_iot23_labels(self, label, detail, expected):
        assert map_behaviour(label, detail) == expected


class TestParser:
    def test_roundtrip(self, tmp_path, flows):
        p = tmp_path / 'conn.log.labeled'
        write_conn_log(flows.drop(columns=['behaviour']), p)
        df = parse_conn_log(p)
        assert len(df) == len(flows)
        assert list(df['behaviour'].value_counts().sort_index()) == \
            list(flows['behaviour'].value_counts().sort_index())
        assert df['orig_bytes'].dtype.kind in 'if'

    def test_iot23_space_separated_tail(self):
        text = (
            "#separator \\x09\n#path\tconn\n"
            "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\t"
            "duration\torig_bytes\tresp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\t"
            "history\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes\t"
            "tunnel_parents   label   detailed-label\n"
            "1.0\tCx\t1.1.1.1\t1000\t2.2.2.2\t23\ttcp\t-\t-\t-\t-\tS0\t-\t-\t0\tS\t1\t40\t0\t0\t"
            "-   Malicious   PartOfAHorizontalPortScan\n"
        )
        df = parse_conn_log(text.encode())
        assert df.loc[0, 'behaviour'] == 'PortScan'
        assert df.loc[0, 'id.resp_p'] == 23
        assert np.isnan(df.loc[0, 'duration'])

    def test_unlabelled_has_no_behaviour(self, tmp_path, flows):
        p = tmp_path / 'live.conn.log'
        write_conn_log(flows.drop(columns=['behaviour']), p, labelled=False)
        df = parse_conn_log(p)
        assert 'behaviour' not in df.columns

    def test_reservoir_sampling_caps_rows(self, tmp_path, flows):
        p = tmp_path / 'c.log'
        write_conn_log(flows.drop(columns=['behaviour']), p)
        assert len(parse_conn_log(p, max_rows=100)) == 100

    def test_empty_file_raises(self):
        with pytest.raises(ValueError):
            parse_conn_log(b"#fields\tts\n")

    def test_load_multiple_tags_capture(self, tmp_path, flows):
        for name in ('capA', 'capB'):
            (tmp_path / name).mkdir()
            write_conn_log(flows.head(50).drop(columns=['behaviour']), tmp_path / name / 'conn.log.labeled')
        df = load_conn_logs(sorted(tmp_path.rglob('conn.log.labeled')))
        assert set(df['capture']) == {'capA', 'capB'} and len(df) == 100


class TestFeatures:
    def test_shape_and_no_nans(self, flows):
        X = extract_flow_features(flows)
        assert list(X.columns) == feature_names()
        assert len(X) == len(flows)
        assert not X.isna().any().any()

    def test_no_ip_features(self):
        assert not any('orig_h' in f or 'resp_h' in f for f in feature_names())

    def test_port_flags(self):
        df = pd.DataFrame({'id.resp_p': [502, 1883, 23, 443, 50000], 'proto': ['tcp'] * 5})
        X = extract_flow_features(df)
        assert X['port_modbus'].tolist() == [1, 0, 0, 0, 0]
        assert X['port_mqtt'].tolist() == [0, 1, 0, 0, 0]
        assert X['port_telnet'].tolist() == [0, 0, 1, 0, 0]
        assert X['port_ephemeral'].tolist() == [0, 0, 0, 0, 1]


class TestDetector:
    def test_untrained_raises(self, flows):
        with pytest.raises(RuntimeError):
            DynamicDetector().predict_proba_df(flows)

    def test_single_class_raises(self, flows):
        benign = flows[flows['behaviour'] == 'Benign']
        with pytest.raises(ValueError):
            DynamicDetector(n_estimators=5).fit(benign)

    def test_learns_synthetic_behaviours(self, trained):
        test = generate_flows({'Benign': 200, 'PortScan': 50, 'DDoS': 30}, seed=99)
        test['behaviour'] = [map_behaviour(a, b) for a, b in zip(test['label'], test['detailed-label'])]
        m = trained.evaluate(test)
        assert m['accuracy'] > 0.9
        assert m['binary']['false_positive_rate'] < 0.05
        assert m['binary']['detection_rate'] > 0.9

    def test_benign_capture_verdict(self, trained):
        rep = trained.analyse_capture(generate_flows({'Benign': 200}, seed=11))
        assert rep.verdict == 'BENIGN'
        assert rep.capture_score < 0.5
        assert rep.behaviour_profile == {}

    def test_malicious_capture_verdict_and_mitre(self, trained):
        cap = generate_flows({'Benign': 200, 'PortScan': 40}, seed=12)
        db = load_behaviour_mitre(config.MITRE_BEHAVIOUR_JSON_PATH)
        rep = trained.analyse_capture(cap, mitre_db=db)
        assert rep.verdict == 'MALICIOUS'
        assert rep.dominant_behaviour == 'PortScan'
        assert any(t['id'] == 'T0846' for t in rep.mitre['techniques'])
        assert rep.explanation, "flagged flows should come with an explanation"
        rep.to_dict()  # serialisable structure

    def test_feature_importance_sorted(self, trained):
        imp = trained.feature_importance()
        vals = [r['importance'] for r in imp]
        assert vals == sorted(vals, reverse=True)
        assert abs(sum(vals) - 1.0) < 1e-6

    def test_save_load_roundtrip(self, trained, flows, tmp_path):
        path = tmp_path / 'dyn.joblib'
        trained.save(path, metadata={'data_source': 'synthetic'})
        loaded = DynamicDetector.load(path)
        a = trained.predict_proba_df(flows.head(30))['malicious_prob'].values
        b = loaded.predict_proba_df(flows.head(30))['malicious_prob'].values
        assert np.allclose(a, b)
        assert loaded.metadata['data_source'] == 'synthetic'


class TestBehaviourMitre:
    def test_mapping_file_covers_all_malicious_classes(self):
        db = load_behaviour_mitre(config.MITRE_BEHAVIOUR_JSON_PATH)
        for cls in ['PortScan', 'C&C', 'DDoS', 'Attack', 'FileDownload', 'Okiru']:
            assert cls in db and db[cls]['techniques']
            for t in db[cls]['techniques']:
                assert t['id'].startswith('T0')  # ICS matrix IDs

    def test_merge_deduplicates(self):
        db = load_behaviour_mitre(config.MITRE_BEHAVIOUR_JSON_PATH)
        m = merge_mitre(['PortScan', 'Okiru'], db)
        ids = [t['id'] for t in m['techniques']]
        assert len(ids) == len(set(ids)) and 'T0846' in ids
