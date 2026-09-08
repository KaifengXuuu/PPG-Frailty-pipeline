"""Launch the local V5 Dash control panel."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'src'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from ppg_frailty.dashboard import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run the local PPG Frailty V5 UI.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in 1..65535')
    if args.host not in {'127.0.0.1', 'localhost', '::1'}:
        parser.error('dashboard is local-only; use a loopback --host')
    app = create_app(ROOT)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
