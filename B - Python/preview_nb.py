import json

nb_path = r"C:\Users\asherk\PycharmProjects\DS_Training\B - Python\pandas\pandas_puzzles_with_solutions.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# Print first 15 markdown cells to understand structure
md_count = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown' and md_count < 15:
        src = ''.join(cell.get('source', []))
        print(f'\n[{i}] Markdown cell:')
        print(src[:150])
        md_count += 1
