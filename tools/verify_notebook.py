"""Compile all Python cells, check the embedded source and optionally nbformat schema."""
import ast
import base64
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
count = 0
notebooks = [path for path in ROOT.rglob('*.ipynb')
             if not any(part.startswith('.') for part in path.relative_to(ROOT).parts)]
assert notebooks == [ROOT/'private_3d_studio.ipynb'], 'Keep one canonical notebook: private_3d_studio.ipynb'
for path in notebooks:
    nb = json.loads(path.read_text())
    assert nb['nbformat'] == 4
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            ast.parse(source, filename=f'{path.name}:cell{i}')
            count += 1
            if 'payload = json.loads(zlib.decompress' in source:
                tree = ast.parse(source)
                payload_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                    and n.func.attr == 'b85decode')
                payload = json.loads(zlib.decompress(base64.b85decode(ast.literal_eval(payload_node.args[0]))))
                for rel, text in payload.items():
                    assert text == (ROOT/rel).read_text(encoding='utf-8-sig'), rel
    try:
        import nbformat
    except ImportError:
        print('nbformat unavailable: schema validation not executed')
    else:
        nbformat.validate(nb)
    print('Validated cell syntax and embedded source:', path.name)
print('Python cells compiled:', count)
