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

# prepare header imports
header = '''import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
print('--- Begin executing cells 77..100 ---')
'''

code_blocks = [header]

for cell in nb['cells'][start_idx+1:]:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        # skip empty cells
        if src.strip():
            code_blocks.append('\n# --- cell break ---\n')
            code_blocks.append(src)

assembled = '\n'.join(code_blocks)

# Wrap execution to continue on exceptions but show traceback
runner = f"""
try:
{assembled}
except Exception as e:
    print('ERROR during execution:')
    traceback.print_exc()
    raise
"""

with open('C:\\Users\\asherk\\PycharmProjects\\DS_Training\\B - Python\\nb_run_from_77_exec_run.py', 'w', encoding='utf-8') as f:
    f.write(runner)

print('Test runner written to nb_run_from_77_exec_run.py')
