import json

nb_path = r"C:\Users\asherk\PycharmProjects\DS_Training\B - Python\pandas\pandas_puzzles_with_solutions.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# Find all problem numbers and their code cells
problem_num = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown':
        src = ''.join(cell.get('source', []))
        # Match **N.** pattern
        if src.startswith('**') and src[2].isdigit():
            problem_num = int(src.split('**')[1].split('.')[0])
            if problem_num == 24:
                print(f'Exercise 24 at markdown cell {i}:')
                print(src[:200])
                # Find the next code cell
                for j in range(i+1, min(i+5, len(nb['cells']))):
                    if nb['cells'][j].get('cell_type') == 'code':
                        code_src = ''.join(nb['cells'][j].get('source', []))
                        print(f'\nCode cell {j}:')
                        print(code_src[:300])
                        # Check for errors
                        outputs = nb['cells'][j].get('outputs', [])
                        if outputs:
                            print(f'\nOutputs: {len(outputs)} items')
                            for out in outputs:
                                if out.get('output_type') == 'error':
                                    print(f'ERROR: {out.get("ename")}')
                                    print(f'  {out.get("evalue")}')
                        break
