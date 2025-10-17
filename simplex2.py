import numpy as np

np.set_printoptions(suppress=True, precision=8)

def prompt_list(prompt, n_expected=None):
    while True:
        try:
            line = input(prompt).strip()
            parts = line.replace(',', ' ').split()
            nums = [float(p) for p in parts]
            if n_expected is not None and len(nums) != n_expected:
                print(f"Ожидалось {n_expected} чисел, получили {len(nums)}. Повторите ввод.")
                continue
            return nums
        except Exception:
            print("Ошибка ввода, повторите. Пример: 1 2 3")

def read_problem():
    print("Ввод исходной задачи (все ограничения вида <=, цель — максимизация).")
    n = int(input("Число переменных n: "))
    m = int(input("Число ограничений m: "))
    print(f"Введите коэффициенты целевой функции (длина {n}):")
    c = np.array(prompt_list("c: ", n), dtype=float)
    print(f"Введите матрицу A построчно ({m} строк, каждая по {n} чисел):")
    A_rows = []
    for i in range(m):
        row = prompt_list(f"  строка {i+1}: ", n)
        A_rows.append(row)
    A = np.array(A_rows, dtype=float)
    print(f"Введите правую часть b (длина {m}):")
    b = np.array(prompt_list("b: ", m), dtype=float)
    if np.any(b < 0):
        raise ValueError("Все значения b должны быть неотрицательны для ограничений вида '<='.")
    return A, b, c

class SimplexWithSlacks:
    def __init__(self, A, b, c, var_prefix='x'):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c_orig = np.array(c, dtype=float)
        self.m, self.n_orig = self.A.shape
        self.I = np.eye(self.m)
        self.A_ext = np.column_stack([self.A, self.I])
        self.n = self.A_ext.shape[1]
        self.c = np.zeros(self.n)
        self.c[:self.n_orig] = self.c_orig
        self.var_names = [f"{var_prefix}{i+1}" for i in range(self.n)]
        self.basis = [self.n_orig + i for i in range(self.m)]
        for i in range(self.m):
            if self.b[i] < 0:
                self.A_ext[i, :] *= -1.0
                self.b[i] *= -1.0

    def pretty_tableau(self, A, b, c, basis, header=None):
        if header:
            print(header)
            print('-' * len(header))
        m, n = A.shape
        names = self.var_names
        basis_set = set(basis)
        display_cols = [j for j in range(n) if j not in basis_set]
        if display_cols:
            colw = max(8, max(len(names[j]) for j in display_cols) + 2)
        else:
            colw = 8
        head = f"{'basis':>8} |"
        for j in display_cols:
            head += f" {names[j]:>{colw}} |"
        head += f" {'b':>{colw}}"
        print(head)
        print('-' * len(head))
        for i in range(m):
            bname = names[basis[i]] if basis[i] >= 0 else f"b{i}"
            line = f"{bname:>8} |"
            for j in display_cols:
                line += f" {A[i,j]:{colw}.6g} |"
            line += f" {b[i]:{colw}.6g}"
            print(line)
        B = A[:, basis]
        B_inv = np.linalg.inv(B)
        cb = c[basis]
        z = float(cb.dot(B_inv).dot(b))
        red = c - cb.dot(B_inv).dot(A)
        print('-' * len(head))
        line = f"{'Ред.':>8} |"
        for j in display_cols:
            line += f" {red[j]:{colw}.6g} |"
        line += f" {z:{colw}.6g}"
        print(line)
        print()

    def solve(self, max_iters=500, show_iters=False):
        A = self.A_ext.copy()
        b = self.b.copy()
        c = self.c.copy()
        basis = self.basis.copy()
        m, n = A.shape
        tol = 1e-12
        for it in range(1, max_iters+1):
            B = A[:, basis]
            B_inv = np.linalg.inv(B)
            xb = B_inv.dot(b)
            cb = c[basis]
            pi = cb.dot(B_inv)
            reduced = c - pi.dot(A)
            if show_iters:
                self.pretty_tableau(A, b, c, basis, header=f"Итерация {it}")
            entering = None
            for j in range(n):
                if reduced[j] > tol:
                    entering = j
                    break
            if entering is None:
                x = np.zeros(n)
                for i in range(m):
                    x[basis[i]] = xb[i]
                z = float(c.dot(x))
                return x, z, basis
            a_j = A[:, entering]
            d = B_inv.dot(a_j)
            ratios = [(xb[i] / d[i], i) for i in range(m) if d[i] > tol]
            if not ratios:
                raise RuntimeError("Решение неограниченно (unbounded).")
            _, leave_row = min(ratios, key=lambda t: (t[0], t[1]))
            basis[leave_row] = entering
        raise RuntimeError("Превышено число итераций")

def build_dual(A, b, c):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    m, n = A.shape
    A_dual = -A.T
    b_dual = -c
    c_dual = -b
    return A_dual, b_dual, c_dual

def print_matrix(label, M):
    print(label)
    for row in M:
        print("  " + "  ".join(f"{v:.6g}" for v in row))

def main():
    try:
        A, b, c = read_problem()
        A_d, b_d, c_d = build_dual(A, b, c)
        print_matrix("A двойственной задачи =", A_d)
        print("Правая часть двойственной задачи (-c):", " ".join(f"{v:.6g}" for v in b_d))
        print("Коэффициенты цели для симплекса (-b):", " ".join(f"{v:.6g}" for v in c_d))
        solver_dual = SimplexWithSlacks(A_d, b_d, c_d)
        y_full, z_rel, basis = solver_dual.solve(show_iters=True)
        y = y_full[:A.shape[0]]
        val_dual_min = -z_rel
        print("\nРешение двойственной задачи:")
        for i, yi in enumerate(y):
            print(f"  y{i+1} = {yi:.8g}")
        print(f"Значение целевой функции (двойственная, минимум): {val_dual_min:.8g}")
    except Exception as e:
        print("\nОшибка:", e)

if __name__ == '__main__':
    main()
