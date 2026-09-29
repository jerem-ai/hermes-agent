#!/usr/bin/env python3
"""Read local usage ledgers into a provenance-labelled report without model calls.

Sources remain separate accounting views. Maestro can mirror native CLI calls;
there is deliberately no sum across views or claim of provider invoice spend.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import tempfile
import time

TOKEN_KEYS = ('input', 'output', 'cacheRead', 'cacheWrite')


def number(value):
    return value if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else 0


def timestamp(value):
    if isinstance(value, (int, float)):
        return value / 1000 if value > 100000000000 else value
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (ValueError, AttributeError, TypeError):
        return 0


def label(value):
    # Never return arbitrary transcript text or terminal control sequences.
    text = str(value or 'unknown')
    return ''.join(c for c in text if c.isprintable())[:120]


def open_db(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    return db


class View:
    def __init__(self, source, basis):
        self.source, self.basis = source, basis
        self.rows, self.coverage, self.warnings = {}, 'available', set()
        self.latest = 0

    def add(self, model='unknown', provider='unknown', billing='unknown', at=0,
            input=0, output=0, cacheRead=0, cacheWrite=0, estimated=None, actual=None, records=1):
        key = tuple(label(v) for v in (model, provider, billing))
        row = self.rows.setdefault(key, dict(zip(('model', 'provider', 'billing'), key)) |
                                   {k: 0 for k in TOKEN_KEYS} | {'records': 0, 'estimatedUsd': None, 'actualUsd': None})
        for field, value in zip(TOKEN_KEYS, (input, output, cacheRead, cacheWrite)):
            row[field] += number(value)
        row['records'] += records
        for field, value in [('estimatedUsd', estimated), ('actualUsd', actual)]:
            if value is not None:
                row[field] = (row[field] or 0) + number(value)
        self.latest = max(self.latest, timestamp(at))

    def report(self):
        return {'source': self.source, 'coverage': self.coverage, 'basis': self.basis,
                'latestAt': self.latest or None, 'warnings': sorted(self.warnings),
                'rows': sorted(self.rows.values(), key=lambda r: -(r['input'] + r['output']))}


def database_views(home, cutoff):
    views = []
    definitions = [
        ('Maestro chat', home / '.maestro/state/maestro.db',
         'Recorded chat calls in the window; cost is a Maestro estimate.',
         'SELECT model,input_tokens,output_tokens,cost_usd,timestamp FROM cost_tracking WHERE timestamp>=?', cutoff * 1000),
        ('Maestro runs', home / '.maestro/state/maestro.db',
         'Lifetime counters for runs started in the window. May overlap native CLI views.',
         'SELECT agent_id,input_tokens,output_tokens,cost_usd,created_at FROM runs WHERE created_at>=?', cutoff * 1000),
        ('Hermes providers', home / '.hermes/state.db',
         'Per-model lifetime counters for sessions started in the window; provider names are recorded billing attribution.',
         'SELECT u.model,u.billing_provider,u.billing_mode,u.input_tokens,u.output_tokens,u.cache_read_tokens,u.cache_write_tokens,u.estimated_cost_usd,u.actual_cost_usd,s.started_at FROM session_model_usage u JOIN sessions s ON s.id=u.session_id WHERE s.started_at>=?', cutoff),
    ]
    for source, path, basis, query, since in definitions:
        view = View(source, basis)
        try:
            with open_db(path) as db:
                for row in db.execute(query, (since,)):
                    row = dict(row)
                    view.add(model=row.get('model', row.get('agent_id')), provider=row.get('billing_provider'),
                             billing=row.get('billing_mode'), at=row.get('timestamp', row.get('created_at', row.get('started_at'))),
                             input=row['input_tokens'], output=row['output_tokens'],
                             cacheRead=row.get('cache_read_tokens', 0), cacheWrite=row.get('cache_write_tokens', 0),
                             estimated=row.get('estimated_cost_usd', row.get('cost_usd')), actual=row.get('actual_cost_usd'))
        except (sqlite3.Error, OSError):
            view.coverage = 'unavailable'
            view.warnings.add('Database or required usage schema unavailable; no migration attempted.')
        views.append(view.report())
    return views


def events(path, deadline, view):
    try:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if time.monotonic() > deadline:
                    view.coverage = 'partial'
                    view.warnings.add('Scan time limit reached; totals cover only scanned records.')
                    return
                # Skip message bodies unless the record carries usage or model metadata.
                if not any(s in line for s in ('"token_count"', '"turn_context"', '"session_meta"', '"usage"')):
                    continue
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
                except (ValueError, TypeError):
                    view.coverage = 'partial'
                    view.warnings.add('Malformed records skipped.')
    except (OSError, UnicodeError):
        view.coverage = 'partial'
        view.warnings.add('Unreadable session file skipped.')


def codex_view(home, cutoff, seconds=30):
    view = View('Codex', 'Usage-event deltas in the window; input includes cached input. Subscription invoice cost is unavailable.')
    root = home / '.codex'
    if not root.exists():
        view.coverage = 'unavailable'
        return view.report()
    paths = set()
    try:
        with open_db(root / 'state_5.sqlite') as db:
            paths.update(Path(r[0]) for r in db.execute('SELECT rollout_path FROM threads WHERE updated_at>=?', (cutoff,)) if r[0])
    except (sqlite3.Error, OSError):
        paths.update(root.glob('sessions/**/*.jsonl'))
        paths.update(root.glob('archived_sessions/**/*.jsonl'))
    deadline = time.monotonic() + seconds
    seen_events = set()
    for path in sorted(paths):
        if time.monotonic() > deadline:
            view.coverage = 'partial'; view.warnings.add('Scan time limit reached; totals cover only scanned records.'); break
        if not path.resolve().is_relative_to(root.resolve()):
            view.coverage = 'partial'; view.warnings.add('Out-of-root rollout path skipped.'); continue
        previous, model, provider, session = {}, 'unknown', 'unknown', None
        for event in events(path, deadline, view):
            payload = event.get('payload') or {}
            if not isinstance(payload, dict):
                continue
            if event.get('type') == 'session_meta':
                session = payload.get('id')
                provider = payload.get('model_provider', 'unknown')
            if event.get('type') == 'turn_context':
                model = payload.get('model', model)
            if event.get('type') != 'event_msg' or payload.get('type') != 'token_count':
                continue
            info = payload.get('info') or {}
            totals = info.get('total_token_usage') if isinstance(info, dict) else None
            if not isinstance(totals, dict):
                continue
            current = {k: number(totals.get(k)) for k in ('input_tokens', 'output_tokens', 'cached_input_tokens', 'cache_write_input_tokens')}
            reset = any(current[k] < previous.get(k, 0) for k in current)
            last = info.get('last_token_usage')
            if (reset or not previous) and isinstance(last, dict):
                delta = {k: number(last.get(k)) for k in current}
            elif reset:
                view.coverage = 'partial'; view.warnings.add('Counter reset without per-call usage skipped.')
                previous = current; continue
            else:
                delta = {k: current[k] - previous.get(k, 0) for k in current}
            previous = current
            identity = (session or str(path), event.get('timestamp'), tuple(current.values()))
            if identity in seen_events:
                continue
            seen_events.add(identity)
            at = timestamp(event.get('timestamp'))
            if at >= cutoff and any(delta.values()):
                view.add(model=model, provider=provider, at=at, input=delta['input_tokens'], output=delta['output_tokens'],
                         cacheRead=delta['cached_input_tokens'], cacheWrite=delta['cache_write_input_tokens'])
    return view.report()


def claude_view(home, cutoff, seconds=30):
    view = View('Claude Code', 'Assistant usage records in the window, deduplicated by request/message ID. Input includes cache reads and writes. Invoice cost is unavailable.')
    root = home / '.claude/projects'
    if not root.exists():
        view.coverage = 'unavailable'
        return view.report()
    deadline = time.monotonic() + seconds
    messages = {}
    for path in sorted(root.glob('**/*.jsonl')):
        if time.monotonic() > deadline:
            view.coverage = 'partial'; view.warnings.add('Scan time limit reached; totals cover only scanned records.'); break
        try:
            if path.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        for event in events(path, deadline, view):
            message = event.get('message') or {}
            if event.get('type') != 'assistant' or not isinstance(message, dict):
                continue
            usage = message.get('usage')
            if not isinstance(usage, dict) or timestamp(event.get('timestamp')) < cutoff:
                continue
            identity = (event.get('requestId'), message.get('id'))
            if not any(identity):
                view.coverage = 'partial'; view.warnings.add('Usage without stable identity skipped.'); continue
            row = messages.setdefault(identity, {'model': message.get('model'), 'at': event.get('timestamp'),
                                                'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0})
            for field, key in [('input', 'input_tokens'), ('output', 'output_tokens'), ('cacheRead', 'cache_read_input_tokens'), ('cacheWrite', 'cache_creation_input_tokens')]:
                row[field] = max(row[field], number(usage.get(key)))
    for row in messages.values():
        row['input'] += row['cacheRead'] + row['cacheWrite']
        view.add(provider='unknown', **row)
    return view.report()


def collect(home, days=30, seconds=30):
    if not 1 <= days <= 366:
        raise ValueError('days must be between 1 and 366')
    now = time.time(); cutoff = now - days * 86400
    return {'schemaVersion': 1, 'generatedAt': now, 'days': days, 'modelCalls': 0,
            'hosts': [{'host': socket.gethostname(), 'views': database_views(home, cutoff) +
                       [codex_view(home, cutoff, seconds), claude_view(home, cutoff, seconds)]}],
            'accounting': 'Views may overlap. Do not sum across sources or hosts. Unknown cost is not zero; estimates are not invoices.'}


def render(report):
    lines = [f"Combined usage: last {report['days']} days", report['accounting']]
    for host in report['hosts']:
        lines.append('\nHost: ' + label(host['host']))
        for view in host['views']:
            lines.extend([f"\n{view['source']} [{view['coverage']}]", view['basis'],
                          'Model | Provider | Billing | Records | Input | Output | Cache read | Cache write | Estimated USD | Recorded USD'])
            for row in view['rows'][:20]:
                cost = lambda value: 'unknown' if value is None else f'{value:.4f}'
                lines.append(' | '.join([label(row['model']), label(row['provider']), label(row['billing']), str(row['records']),
                                       *[str(row[k]) for k in TOKEN_KEYS], cost(row['estimatedUsd']), cost(row['actualUsd'])]))
            if len(view['rows']) > 20:
                lines.append(f"{len(view['rows']) - 20} more model rows in the JSON report.")
            if not view['rows']:
                lines.append('No recorded usage in this view.' if view['coverage'] == 'available' else 'Usage unavailable.')
            lines.extend('Coverage: ' + warning for warning in view['warnings'])
    return '\n'.join(lines)


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.usage-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(report, stream, indent=2); stream.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--home', type=Path, default=Path.home())
    parser.add_argument('--scan-seconds', type=int, default=30)
    parser.add_argument('--scout', action='store_true', help='Also read Scout using the registered mac-mini SSH alias')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.scan_seconds <= 300:
        parser.error('--scan-seconds must be between 1 and 300')
    report = collect(args.home, args.days, args.scan_seconds)
    if args.scout:
        try:
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'mac-mini',
                       'python3 - --json --days ' + str(args.days) + ' --scan-seconds ' + str(args.scan_seconds)]
            result = subprocess.run(command, input=Path(__file__).read_text(), text=True, capture_output=True,
                                    check=True, timeout=args.scan_seconds * 2 + 30)
            remote = json.loads(result.stdout)
            report['hosts'].extend(remote['hosts'])
        except (OSError, subprocess.SubprocessError, ValueError):
            report['hosts'].append({'host': 'Scout', 'views': [{'source': 'Scout usage', 'coverage': 'unavailable',
                                   'basis': 'Remote read failed; no model request was sent.', 'rows': [], 'warnings': []}]})
    if args.output:
        save(args.output, report)
    print(json.dumps(report) if args.json else render(report))


if __name__ == '__main__':
    main()
