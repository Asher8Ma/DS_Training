import json

nb_path = r"C:\Users\asherk\PycharmProjects\DS_Training\B - Python\pandas\pandas_puzzles_with_solutions.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# Find markdown cells with exercise numbers
exercises = {}
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown':
        src = ''.join(cell.get('source', []))
        for ex_num in range(1, 30):
            if f'## {ex_num}.' in src or f'# {ex_num}.' in src:
                exercises[ex_num] = i
                print(f'Exercise {ex_num}: cell index {i}')
                print(f'  Preview: {src[:100]}...')

print(f'\nExercises found: {sorted(exercises.keys())}')
