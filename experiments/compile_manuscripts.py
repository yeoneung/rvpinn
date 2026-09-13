"""Build both paper PDFs and fail on any LaTeX/BibTeX error."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ('main', 'supplement'):
        commands = [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'{name}.tex'],
                    ['bibtex', name]]
        commands += [commands[0]] * 2
        for command in commands:
            subprocess.run(command, cwd=ROOT / 'manuscript', check=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments' / 'audit_manuscript_format.py')],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
