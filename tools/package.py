"""Package source without runtime data, credentials or local test dependencies."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT.parent/'private-3d-studio.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and not any((p.startswith('.') and p != '.gitignore') or p == '__pycache__' for p in path.relative_to(ROOT).parts):
            z.write(path, Path(ROOT.name)/path.relative_to(ROOT))
print(target)
print('Bytes:', target.stat().st_size)
