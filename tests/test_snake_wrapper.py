"""Regression : `./snake` ne doit pas exiger uv (issue #55).

Bug historique : le wrapper faisait `exec uv run ...` sans condition. Sur une
machine sans uv, le programme ne demarrait pas du tout, avant meme le parsing
des arguments. Le wrapper retombe desormais sur un venv local (.venv/, venv/)
puis sur python3 avec src/ dans PYTHONPATH, les chemins etant resolus par
rapport au repertoire du script et non au repertoire courant.
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAKE = ROOT / "snake"


def _run(argv, env, cwd=None):
    return subprocess.run(
        [str(SNAKE), *argv], env=env, cwd=cwd,
        capture_output=True, text=True, timeout=120,
    )


def _env_without_uv():
    """Environnement minimal : ni uv, ni PYTHONPATH herite."""
    env = {"PATH": "/usr/bin:/bin", "SDL_VIDEODRIVER": "dummy"}
    for key in ("HOME", "LANG", "LC_ALL"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def test_wrapper_runs_without_uv_on_path():
    result = _run(["-sessions", "1", "-visual", "off"], _env_without_uv())
    assert result.returncode == 0, result.stderr
    assert "Game over, max length" in result.stdout


def test_wrapper_resolves_paths_relative_to_script(tmp_path):
    result = _run(["-sessions", "1", "-visual", "off"], _env_without_uv(),
                  cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Game over, max length" in result.stdout


def test_wrapper_fails_cleanly_without_any_interpreter(tmp_path):
    # Copie isolee : aucun venv local, et un PATH sans uv ni python3.
    (tmp_path / "src").mkdir()
    wrapper = tmp_path / "snake"
    wrapper.write_bytes(SNAKE.read_bytes())
    wrapper.chmod(0o755)
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    result = subprocess.run(
        [str(wrapper), "-sessions", "1"], env={"PATH": str(empty_bin)},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 127
    assert "aucun interpreteur Python" in result.stderr
