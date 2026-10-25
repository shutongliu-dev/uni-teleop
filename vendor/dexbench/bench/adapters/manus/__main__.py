"""Record raw/normalized human motion. Never initializes or commands hardware/physics."""
import argparse
import json
from pathlib import Path
import sys
import time
from pydantic import ValidationError
from .recording import CaptureRecorder, DEFAULT_ROOT
from .schema import CaptureConfig
from .sources import host_clock_id, mock_packets, packets_from_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--mock', type=int, metavar='FRAMES')
    source.add_argument('--replay', type=Path, help='acquisition.jsonl; records a new derivation with original timestamps')
    source.add_argument('--stdin', action='store_true', help='JSONL from an isolated licensed SDK process')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.mock is not None and args.mock < 1:
        parser.error('--mock must be positive')
    clock = host_clock_id()
    if args.config:
        config = CaptureConfig.model_validate_json(args.config.read_text())
    elif args.mock is not None:
        config = CaptureConfig(calibration_id='synthetic_identity', target_clock_id=clock)
    else:
        parser.error('--config is required for replay and stdin')
    recorder = CaptureRecorder(config, args.output_root, replay_of=str(args.replay) if args.replay else None)
    replay_stream = None
    reason = 'source_finished'
    try:
        if args.mock is not None:
            source_packets = mock_packets(args.mock, clock_id=config.target_clock_id)
        else:
            replay_stream = args.replay.open() if args.replay else sys.stdin
            source_packets = packets_from_jsonl(replay_stream)
        for packet in source_packets:
            # Replay keeps original source_kind and source timestamps. It is record-only;
            # stale data is retained with false control validity, never made fresh silently.
            now = packet.capture_time_ns if args.mock is not None else time.monotonic_ns()
            recorder.ingest(packet, now)
    except (ValidationError, ValueError) as exc:
        reason = 'invalid_input'
        recorder.invalid(type(exc).__name__)  # Do not echo arbitrary input/credentials.
        raise SystemExit('Invalid acquisition input; capture retained with an error event.') from None
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception:
        reason = 'collector_error'
        raise
    finally:
        if replay_stream is not None and replay_stream is not sys.stdin:
            replay_stream.close()
        recorder.close(reason)
        print(json.dumps({'capture': str(recorder.path), 'counts': recorder.counts, 'termination': reason}))

if __name__ == '__main__':
    main()
