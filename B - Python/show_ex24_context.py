import json
import re

nb_path = r"C:\Users\asherk\PycharmProjects\DS_Training\B - Python\pandas\pandas_puzzles_with_solutions.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# Find exercise 24 code cell and surrounding cells
for i, cell in enumerate(nb['cells']):
    if i >= 58 and i <= 65:
        if cell.get('cell_type') == 'markdown':
            src = ''.join(cell.get('source', []))
            print(f'[{i}] MARKDOWN: {src[:100]}')
        elif cell.get('cell_type') == 'code':
            src = ''.join(cell.get('source', []))
            print(f'[{i}] CODE:\n{src}')
            if i == 60:  # This is ex24
                print('\n*** THIS IS EX24 - NEEDS FIX ***\n')
