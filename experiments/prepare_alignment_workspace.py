"""Create an isolated, recoverable re-evaluation workspace; never overwrite it."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deployment_numerics import DEPLOYMENT_VERSION
from src.deployed_policy import ACTION_SEARCH_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='alignment_run')
    args = parser.parse_args()
    target = (ROOT / args.target).resolve()
    if target.parent != ROOT.resolve():
        raise ValueError('target must be a new direct child of the project directory')
    if target.exists():
        raise FileExistsError(f"inspect or resume the existing workspace: {target}")
    target.mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.log', '*.synctex*')
    for name in ('src', 'tests', 'scripts', 'configs', 'data', 'checkpoints',
                 'results', 'manuscript'):
        shutil.copytree(ROOT / name, target / name, ignore=ignore)
    (target / 'experiments' / 'results').mkdir(parents=True)
    (target / 'experiments' / 'logs').mkdir()
    for path in (ROOT / 'experiments').iterdir():
        if path.is_file() and path.suffix in ('.py', '.ps1'):
            shutil.copy2(path, target / 'experiments' / path.name)
    for name in ('Makefile', 'README.md', 'environment.yml', 'requirements.txt',
                 'REPRODUCIBILITY.md', 'HARDWARE_SOFTWARE.md'):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, target / name)
    marker = {'source': str(ROOT), 'deployment_version': DEPLOYMENT_VERSION,
              'action_search_version': ACTION_SEARCH_VERSION,
              'purpose': 'isolated numerical re-evaluation; not a submission release'}
    (target / 'ALIGNMENT_WORKSPACE.json').write_text(json.dumps(marker, indent=2),
                                                    encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
