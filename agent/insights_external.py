"""Render an opt-in local usage snapshot. Never scans transcripts or calls models."""
import json
from pathlib import Path
import time


def read_combined_usage(days, source=None, path=None, now=None):
    path = path or Path.home() / '.maestro/state/combined-usage.json'
    if source:
        return {'status': 'filtered', 'note': 'Combined usage is omitted with a Hermes platform filter.'}
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('oversize')
        report = json.loads(path.read_text())
        if report.get('schemaVersion') != 1 or not isinstance(report.get('hosts'), list):
            raise ValueError('schema')
        age = (time.time() if now is None else now) - report['generatedAt']
        if age < -60 or age > 3600:
            return {'status': 'stale', 'note': 'Combined usage snapshot is stale. Refresh it before comparing usage.'}
        if report.get('days') != days:
            return {'status': 'window_mismatch', 'note': f'Combined snapshot covers {report.get("days")} days; this request covers {days}.'}
        return {'status': 'available', 'report': report}
    except (OSError, ValueError, KeyError, TypeError):
        return {'status': 'unavailable', 'note': 'Combined usage snapshot is unavailable.'}


def format_combined_usage(value):
    if not value:
        return ''
    if value.get('status') != 'available':
        return '\n\nCombined usage: ' + value.get('note', 'unavailable')
    try:
        # The collector is a standard-library module with no import-time effects.
        from agent.combined_usage import render
        return '\n\n' + render(value['report'])
    except (ValueError, KeyError, TypeError, AttributeError):
        return '\n\nCombined usage: invalid snapshot; refresh required.'
