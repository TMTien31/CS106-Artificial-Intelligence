from pathlib import Path
from ortools.algorithms.python import knapsack_solver
import time

def read_data(kp_file):
    arr = {'weight': [], 'values': []}
    with open(kp_file, 'rb') as file:
        data = file.read()
    lines = data.split(b'\n')
    lines = [line.decode('utf-8') for line in lines if line.strip()]
    n = 0
    c = 0
    for line in lines:
        tokens = list(map(int, line.strip().split()))
        if len(tokens) == 1:
            if n == 0:
                n = tokens[0]
            else:
                c = tokens[0]
        elif len(tokens) == 2:
            arr['values'].append(tokens[0])
            arr['weight'].append(tokens[1])
    return arr['values'], [arr['weight']], [c]

def run_tests():
    base_path = Path(__file__).resolve().parent / "kplib"
    #group_folders = sorted(base_path.glob('02*')) #chọn nhóm lẻ
    group_folders = sorted(base_path.glob('[0-9][0-9]*'))  # chạy toàn bộ
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample"
    )
    solver.set_time_limit(180)  # 3 phút
    output_path = Path(__file__).resolve().parent / "result.txt"

    with open(output_path, 'a', encoding='utf-8') as f:
        for group_path in group_folders:
            f.write(f'GROUP: {group_path.name}\n\n')
            print(f'===> Running group: {group_path.name}')
            
            subfolders = sorted(group_path.glob('n*'))[:5]  # chọn 5 kích thước khác nhau
            for subfolder in subfolders:
                test_case_folder = subfolder / "R01000"  # cố định dùng R01000
                kp_file = test_case_folder / "s000.kp"   # dùng file s000.kp
                if not kp_file.exists():
                    f.write(f"File {kp_file} not found.\n\n")
                    continue

                print(f'  -> Running {kp_file.relative_to(base_path)}...')
                values, weights, capacities = read_data(kp_file)
                t1 = time.time()
                solver.init(values, weights, capacities)
                computed_value = solver.solve()
                t2 = time.time()
                is_optimal = solver.is_solution_optimal()

                packed_items = []
                packed_weights = []
                total_weight = 0

                for i in range(len(values)):
                    if solver.best_solution_contains(i):
                        packed_items.append(i)
                        packed_weights.append(weights[0][i])
                        total_weight += weights[0][i]

                # Ghi kết quả
                status = "[Optimal]" if is_optimal else "[Not Optimal]"
                f.write(f'--- TEST CASE: {kp_file.relative_to(base_path)} ---\n')
                f.write(f'Status: {status}\n')
                f.write(f'Total value = {computed_value}\n')
                f.write(f'Total weight: {total_weight}\n')
                f.write(f'Number of items packed = {len(packed_items)}\n')
                f.write(f'Run time = {(t2 - t1) * 1000:.2f} ms\n')  # <-- đổi sang ms
                f.write('--------------------------------------------\n\n')


if __name__ == "__main__":
    run_tests()
