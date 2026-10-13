#!/usr/bin/env python3
"""
Write a synthetic Zeek conn.log for demos and testing of the Dynamic Analysis page.

Examples
  python scripts/generate_flows.py --out demo_benign.conn.log  --benign 300
  python scripts/generate_flows.py --out demo_mirai.conn.log   --benign 200 --portscan 80 --okiru 40 --cc 10
  python scripts/generate_flows.py --out demo_ddos.conn.log    --benign 150 --ddos 60 --unlabelled

The output is SYNTHETIC traffic. It is useful to exercise the pipeline but is
not evidence of real-world detection performance.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--benign', type=int, default=200)
    p.add_argument('--portscan', type=int, default=0)
    p.add_argument('--cc', type=int, default=0)
    p.add_argument('--ddos', type=int, default=0)
    p.add_argument('--attack', type=int, default=0)
    p.add_argument('--filedownload', type=int, default=0)
    p.add_argument('--okiru', type=int, default=0)
    p.add_argument('--unlabelled', action='store_true',
                   help='Omit label columns (what a live Digital Twin capture looks like)')
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()

    from modules.dynamic.synthetic import generate_flows
    from modules.dynamic.zeek_parser import write_conn_log
    counts = {'Benign': a.benign, 'PortScan': a.portscan, 'C&C': a.cc, 'DDoS': a.ddos,
              'Attack': a.attack, 'FileDownload': a.filedownload, 'Okiru': a.okiru}
    df = generate_flows({k: v for k, v in counts.items() if v}, seed=a.seed)
    write_conn_log(df, a.out, labelled=not a.unlabelled)
    print(f"Wrote {len(df)} synthetic flows → {a.out}")


if __name__ == '__main__':
    main()
