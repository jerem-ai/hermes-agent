import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/ci/pr_merge_base.sh'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def test_runtime_target_accepts_its_child_and_refuses_unrelated_main(tmp_path):
    git(tmp_path, 'init', '-q', '-b', 'main')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    git(tmp_path, 'commit', '-q', '--allow-empty', '-m', 'unrelated upstream')
    git(tmp_path, 'update-ref', 'refs/remotes/origin/main', 'HEAD')
    git(tmp_path, 'checkout', '-q', '--orphan', 'runtime/研究')
    git(tmp_path, 'commit', '-q', '--allow-empty', '-m', 'preserved runtime root')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    git(tmp_path, 'update-ref', 'refs/remotes/origin/runtime/研究', 'HEAD')
    git(tmp_path, 'checkout', '-q', '-b', 'vision')
    git(tmp_path, 'commit', '-q', '--allow-empty', '-m', 'vision support')
    accepted = subprocess.run(['bash', str(SCRIPT), 'runtime/研究'], cwd=tmp_path, capture_output=True, text=True)
    assert accepted.returncode == 0 and accepted.stdout.strip() == base
    for refused in ('main', 'missing', 'bad ref', '$(touch injected)'):
        result = subprocess.run(['bash', str(SCRIPT), refused], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode != 0
    assert not (tmp_path / 'injected').exists()
