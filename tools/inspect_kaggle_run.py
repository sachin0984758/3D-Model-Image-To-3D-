"""Read saved execution outputs; never execute notebook code or infer success from source."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def inspect(path):
    path = Path(path)
    raw = path.read_bytes()
    notebook = json.loads(raw)
    text = []
    errors = []
    for index, cell in enumerate(notebook.get('cells', [])):
        for output in cell.get('outputs', []):
            value = output.get('text', [])
            text.append(value if isinstance(value, str) else ''.join(value))
            plain = output.get('data', {}).get('text/plain', [])
            text.append(plain if isinstance(plain, str) else ''.join(plain))
            if output.get('output_type') == 'error':
                errors.append(dict(cell=index, name=output.get('ename'), message=output.get('evalue')))
    rendered = re.sub(r'\x1b\[[0-9;]*m', '', '\n'.join(text))
    decoder = json.JSONDecoder()
    models = {}
    for match in re.finditer(r'[\[{]', rendered):
        try:
            value, _ = decoder.raw_decode(rendered, match.start())
        except (ValueError, RecursionError):
            continue
        candidates = value if isinstance(value, list) else [value]
        for item in candidates:
            if (isinstance(item, dict) and item.get('checkpoint') == 'tencent/Hunyuan3D-2.1'
                    and isinstance(item.get('shape'), dict) and isinstance(item.get('paint'), dict)
                    and isinstance(item.get('name'), str)):
                models[item['name']] = item
    results = []
    for name, item in sorted(models.items()):
        shape, paint = item['shape'], item['paint']
        sv, pv = shape.get('validation', {}), paint.get('validation', {})
        results.append(dict(name=name, input_filename=item.get('input', {}).get('filename'),
            shape_status=shape.get('status'), paint_status=paint.get('status'),
            reported_validated_geometry=shape.get('status') == 'completed' and sv.get('faces', 0) > 0,
            reported_validated_pbr=paint.get('status') == 'completed' and pv.get('pbr_verified') is True
                and pv.get('embedded_images', 0) > 0,
            shape=shape, paint=paint))
    cpu = re.findall(r'Ran (\d+) tests? in [^\n]+\s+(OK(?: \([^\n]*\))?|FAILED[^\n]*)', rendered)
    return dict(notebook=path.name, sha256=hashlib.sha256(raw).hexdigest(),
        executed_code_cells=sum(c.get('execution_count') is not None for c in notebook.get('cells', []) if c.get('cell_type') == 'code'),
        last_reported_cpu_suite=dict(tests=int(cpu[-1][0]), outcome=cpu[-1][1]) if cpu else None,
        reported_models=results, execution_errors=errors,
        reported_validated_geometry_count=sum(r['reported_validated_geometry'] for r in results),
        reported_validated_pbr_count=sum(r['reported_validated_pbr'] for r in results),
        observation_only_geometry_lines=re.findall(r'\[shape [^\]]+\][^\n]*faces in[^\n]*', rendered),
        limitation='Saved-output analysis only. No GPU inference or independent GLB reload was performed by this reader. '
                   'Missing output is unverified, not proof of failure. Inspect actual GLBs and the live UI separately.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = inspect(args.notebook)
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result+'\n', encoding='utf-8')
    print(result)
