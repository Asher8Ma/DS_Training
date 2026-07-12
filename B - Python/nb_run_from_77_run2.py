import json
import traceback
import sys

nb_path = r"C:\Users\asherk\PycharmProjects\DS_Training\B - Python\numpy\100_Numpy_exercises.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# find markdown cell with exercise 77 header
start_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown':
        src = ''.join(cell.get('source', []))
        if '#### 77.' in src or '#### 77' in src:
            start_idx = i
            break

if start_idx is None:
    print('Could not find exercise 77 start marker')
    sys.exit(1)

# prepare execution environment with imports
env = {}
exec("import numpy as np\nfrom numpy.lib.stride_tricks import sliding_window_view\nimport warnings\nwarnings.filterwarnings('ignore')\nnp.random.seed(42)\nprint('--- Begin executing cells 77..100 ---')\n", env)

had_error = False
for cell in nb['cells'][start_idx+1:]:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if not src.strip():
            continue
        print('\n--- Executing cell ---\n')
        print(src)
        try:
            exec(src, env)
        except Exception:
            had_error = True
            print('Error executing cell:')
            traceback.print_exc()

if had_error:
    print('\nOne or more cells failed.')
    sys.exit(2)
else:
    print('\nAll cells executed without uncaught exceptions.')
