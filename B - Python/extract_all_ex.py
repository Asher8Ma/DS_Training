import json

nb_path = r"C:\Users\asherk\PycharmProjects\DS_Training\B - Python\pandas\pandas_puzzles_with_solutions.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# Extract all exercises and their code
exercises = {}
problem_num = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown':
        src = ''.join(cell.get('source', []))
        if src.startswith('**') and '.' in src:
            try:
                problem_num = int(src.split('**')[1].split('.')[0])
                exercises[problem_num] = {'md_idx': i, 'md': src[:100], 'code_idx': None}
            except:
                pass
    elif cell.get('cell_type') == 'code' and problem_num > 0:
        if exercises[problem_num]['code_idx'] is None:
            code = ''.join(cell.get('source', []))
            exercises[problem_num]['code_idx'] = i
            exercises[problem_num]['code'] = code

print(f'Found {len(exercises)} exercises')
print(f'Exercises: {sorted(exercises.keys())}')

# Show exercises with issues
for ex_num in [24]:
    if ex_num in exercises:
        ex = exercises[ex_num]
        print(f'\nExercise {ex_num}:')
        print(f'  Code: {ex["code"][:200]}')
