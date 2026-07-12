import json

nb_path = r"C:\Users\asherk\PycharmProjects\DS_Training\B - Python\pandas\pandas_puzzles_with_solutions.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')

# Find exercise 24
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown':
        src = ''.join(cell.get('source', []))
        if '#24' in src or 'Exercise 24' in src or '## 24' in src:
            print(f'Found ex24 marker at index {i}')
            print(f'Content: {src[:200]}')
            break

# Count code cells
code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
print(f'Total code cells: {len(code_cells)}')
