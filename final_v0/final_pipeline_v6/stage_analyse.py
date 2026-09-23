"""CLI equivalent of a notebook-style Dash Analyse action (never trains)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Optional YAML; omitted means function defaults.')
    parser.add_argument('--stage', choices=('input', 'ppg', 'imu', 'motion', 'quality', 'denoiser',
                                           'features', 'representation', 'model', 'aggregation'), default='ppg')
    parser.add_argument('--record-id', default='', help='Manifest recording or file_id from --selections.')
    parser.add_argument('--set', action='append', default=[], metavar='PATH=VALUE', help='Override current YAML/default parameters.')
    parser.add_argument('--selections', default='{}', help='JSON object: files, record_ids, participant_id, '
                        'calibration_path, sqi_artifact, motion_bundle, model_bundle/model_export/model_case.')
    parser.add_argument('--start-s', type=float, default=0.0)
    parser.add_argument('--duration-s', type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    import yaml
    from ppg_frailty.config import _materialize_v2_defaults
    from ppg_frailty.dashboard.workflow_controls import default_configuration, apply_control_values
    from ppg_frailty.dashboard.workflow_service import WorkflowService
    from ppg_frailty.v5.configuration import parse_assignment
    args = build_parser().parse_args(argv)
    try:
        config = yaml.safe_load(args.config.read_text(encoding='utf-8')) if args.config else default_configuration(ROOT)
        if args.config:
            # Presets may omit effective defaults. Use the same materializer as
            # the pipeline, without imposing training-only contracts on a stage
            # exploration (for example raw inputs plus selected feature groups).
            _materialize_v2_defaults(config)
        config = apply_control_values(config, dict(parse_assignment(value) for value in args.set), pipeline_root=ROOT)
        selected = json.loads(args.selections)
        result = WorkflowService(ROOT).analyse(args.stage, config, args.record_id, selections=selected,
                                               start_s=args.start_s, duration_s=args.duration_s)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'error', 'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
