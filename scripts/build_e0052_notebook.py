#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    base = json.loads(Path('local_runs/EXT0006/ext0006-score12-valid.ipynb').read_text())
    oracle = json.loads(Path('local_runs/E0046/e0046-local.ipynb').read_text())
    for original in oracle['cells'][10:19]:
        cell = {'cell_type': original['cell_type'], 'metadata': {},
                'source': original.get('source', [])}
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None; cell['outputs'] = []
        base['cells'].append(cell)
    cell_source = Path('scripts/e0052_mil_cell.py').read_text()
    base['cells'].append({'cell_type':'code','metadata':{},'execution_count':None,
                          'outputs':[], 'source':cell_source.splitlines(keepends=True)})
    base.get('metadata',{}).pop('papermill',None); base.get('metadata',{}).pop('codex',None)
    Path('local_runs/E0052').mkdir(parents=True,exist_ok=True)
    Path('local_runs/E0052/e0052-mil-oracle.ipynb').write_text(json.dumps(base,indent=1)+'\n')
    print('local_runs/E0052/e0052-mil-oracle.ipynb')
if __name__ == '__main__': main()
