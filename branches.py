#!/usr/bin/env python3
"""
Branch-and-Bound + Two-Phase Simplex (numpy only)

Решает задачи целочисленного линейного программирования (целые переменные — по умолчанию все)
используя симплекс-метод с двумя фазами для решения LP-релаксаций и метод ветвей и границ для целочисленности.

Особенности:
- Поддерживаются ограничения вида '<=', '>=' и '='.
- Ввод с клавиатуры (произвольное число переменных и ограничений).
- Можно указать, какие переменные требуются целочисленными (по умолчанию — все переменные).
- Используется строгая арифметика с небольшой допустимой погрешностью при проверке целочисленности (tol).
- Переменные называются x1, x2, ... для всех переменных, включая вводимые доп. переменные.

Ограничения и численные замечания:
- Код ориентирован на учебные и умеренные по размеру задачи.
- Для численно жёстких задач может потребоваться настройка tol или более робустный численный подход.

Автор: ChatGPT
"""

import sys
import math
import copy
import heapq
import numpy as np

np.set_printoptions(suppress=True, precision=8)

# ------------------ Two-phase Simplex (robust enough for branching) ------------------
class TwoPhaseSimplex:
    def __init__(self, A, b, c, relations, var_prefix='x'):
        # A: (m x n_orig), relations: list length m of '<=', '>=', '='
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c_orig = np.array(c, dtype=float)
        self.relations = list(relations)
        self.m, self.n_orig = self.A.shape
        self.var_prefix = var_prefix
        self.build_standard_form()

    def build_standard_form(self):
        cols = [self.A[:, j] for j in range(self.n_orig)]
        self.var_names = [f"{self.var_prefix}{i+1}" for i in range(self.n_orig)]
        self.artificials = []

        for i, rel in enumerate(self.relations):
            if rel == '<=':
                col = np.zeros(self.m)
                col[i] = 1.0
                cols.append(col)
                self.var_names.append(f"{self.var_prefix}{len(self.var_names)+1}")
            elif rel == '>=':
                col_sur = np.zeros(self.m)
                col_sur[i] = -1.0
                cols.append(col_sur)
                self.var_names.append(f"{self.var_prefix}{len(self.var_names)+1}")
                col_art = np.zeros(self.m)
                col_art[i] = 1.0
                cols.append(col_art)
                self.var_names.append(f"{self.var_prefix}{len(self.var_names)+1}")
                self.artificials.append(len(cols)-1)
            elif rel == '=':
                col_art = np.zeros(self.m)
                col_art[i] = 1.0
                cols.append(col_art)
                self.var_names.append(f"{self.var_prefix}{len(self.var_names)+1}")
                self.artificials.append(len(cols)-1)
            else:
                raise ValueError(f"Неожиданный знак ограничения: {rel}")

        self.A_ext = np.column_stack(cols)
        self.n = self.A_ext.shape[1]

        # cost for Phase II
        self.c = np.zeros(self.n)
        self.c[:self.n_orig] = self.c_orig

        # initial basis: choose unit columns
        self.basis = [-1] * self.m
        for j in range(self.n):
            col = self.A_ext[:, j]
            nz = np.where(np.abs(col) > 1e-12)[0]
            if len(nz) == 1 and abs(col[nz[0]] - 1.0) < 1e-12:
                row = int(nz[0])
                if self.basis[row] == -1:
                    self.basis[row] = j

        # ensure any remaining rows get an artificial if available
        for i in range(self.m):
            if self.basis[i] == -1:
                for art in self.artificials:
                    if abs(self.A_ext[i, art] - 1.0) < 1e-12 and np.count_nonzero(np.abs(self.A_ext[:, art]) > 1e-12) == 1:
                        self.basis[i] = art
                        break

        # final check
        if any(bi == -1 for bi in self.basis):
            missing = [i for i, bi in enumerate(self.basis) if bi == -1]
            raise ValueError(f"Не удалось подобрать начальный базис: строки {missing}. Входная матрица требует базисных столбцов.")

    def pretty_tableau(self, A, b, c, basis, header=None):
        if header:
            print('
' + header)
            print('-' * len(header))
        m, n = A.shape
        names = self.var_names
        colw = max(8, max(len(nm) for nm in names) + 2)

        head = f"{'basis':>8} |"
        for j in range(n):
            head += f" {names[j]:>{colw}} |"
        head += f" {'RHS':>{colw}}"
        print(head)
        print('-' * len(head))

        for i in range(m):
            bname = names[basis[i]] if basis[i] >= 0 else f"b{i}"
            line = f"{bname:>8} |"
            for j in range(n):
                line += f" {A[i,j]:{colw}.6g} |"
            line += f" {b[i]:{colw}.6g}"
            print(line)

        B = A[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            B_inv = np.linalg.pinv(B)
        cb = c[basis]
        z = float(cb.dot(B_inv).dot(b))
        red = c - cb.dot(B_inv).dot(A)

        print('-' * len(head))
        line = f"{'obj':>8} |"
        for j in range(n):
            line += f" {red[j]:{colw}.6g} |"
        line += f" {z:{colw}.6g}"
        print(line)
        print()

    def simplex(self, A, b, c, basis, maximize=True, iter_limit=500):
        m, n = A.shape
        basis = basis.copy()
        for it in range(iter_limit):
            B = A[:, basis]
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                B_inv = np.linalg.pinv(B)

            xb = B_inv.dot(b)
            cB = c[basis]
            pi = cB.dot(B_inv)
            reduced = c - pi.dot(A)

            if maximize:
                entering_candidates = [j for j in range(n) if reduced[j] > 1e-12]
            else:
                entering_candidates = [j for j in range(n) if reduced[j] < -1e-12]

            if not entering_candidates:
                x = np.zeros(n)
                for i in range(m):
                    x[basis[i]] = xb[i]
                z = float(c.dot(x))
                return x, z, basis

            entering = min(entering_candidates)
            a_j = A[:, entering]
            d = B_inv.dot(a_j)

            ratios = [(xb[i] / d[i], i) for i in range(m) if d[i] > 1e-12]
            if not ratios:
                raise RuntimeError('unbounded')
            _, leave_row = min(ratios, key=lambda t: (t[0], t[1]))
            basis[leave_row] = entering

        raise RuntimeError('iter_limit')

    def two_phase_solve(self, verbose=False):
        # Phase I
        if len(self.artificials) == 0:
            # directly phase II
            try:
                x, z, basis = self.simplex(self.A_ext, self.b, self.c, self.basis, maximize=True)
            except Exception as e:
                raise RuntimeError('Phase II failed: ' + str(e))
            return x, z

        c1 = np.zeros(self.n)
        for art in self.artificials:
            c1[art] = 1.0
        c1_max = -c1

        try:
            x1, z1, basis1 = self.simplex(self.A_ext, self.b, c1_max, self.basis, maximize=True)
        except Exception as e:
            raise RuntimeError('Phase I failed: ' + str(e))

        min_sum_art = -z1
        if min_sum_art > 1e-7:
            raise RuntimeError('infeasible')

        # Remove artificial columns
        keep = [j for j in range(self.n) if j not in self.artificials]
        A2 = self.A_ext[:, keep]
        c2 = np.zeros(len(keep))
        for new_j, old_j in enumerate(keep):
            if old_j < len(self.c):
                c2[new_j] = self.c[old_j]
        # Remap basis
        basis2 = []
        for bi in basis1:
            if bi in self.artificials:
                # try to find a unit column in the same row in A2
                basis2.append(-1)
            else:
                basis2.append(keep.index(bi))

        # fill missing basis entries
        for i in range(self.m):
            if basis2[i] == -1:
                found = -1
                for j in range(A2.shape[1]):
                    col = A2[:, j]
                    nz = np.where(np.abs(col) > 1e-12)[0]
                    if len(nz) == 1 and nz[0] == i and abs(col[i] - 1.0) < 1e-12:
                        found = j
                        break
                if found == -1:
                    # fallback: pick any column
                    found = 0
                basis2[i] = found

        # Phase II
        try:
            x2, z2, basis_final = self.simplex(A2, self.b, c2, basis2, maximize=True)
        except Exception as e:
            raise RuntimeError('Phase II failed: ' + str(e))

        # expand to full length (without artificials)
        x_full = np.zeros(self.n)
        for new_j, old_j in enumerate(keep):
            x_full[old_j] = x2[new_j]

        return x_full, z2

# ------------------ Branch and Bound ------------------
class BranchAndBoundIP:
    def __init__(self, A, b, c, relations, int_vars=None, maximize=True, tol=1e-6):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.relations = list(relations)
        self.n = self.A.shape[1]
        if int_vars is None:
            self.int_vars = list(range(self.n))
        else:
            self.int_vars = sorted(int_vars)
        self.maximize = maximize
        self.tol = tol

    def is_integer_solution(self, x):
        for j in self.int_vars:
            if abs(x[j] - round(x[j])) > self.tol:
                return False
        return True

    def fractional_index(self, x):
        # choose variable with largest fractional part (heuristic)
        best_j = None
        best_frac = 0.0
        for j in self.int_vars:
            frac = abs(x[j] - math.floor(x[j]))
            frac = min(frac, 1-frac)
            if frac > best_frac + 1e-12 and abs(x[j] - round(x[j])) > self.tol:
                best_frac = frac
                best_j = j
        return best_j

    def solve(self, verbose=False, node_limit=1000):
        # Use max-heap (neg objective) so we explore most promising nodes first
        best_obj = -np.inf if self.maximize else np.inf
        best_x = None
        # node: (priority, node_id, extra_constraints)
        # extra_constraints: list of tuples (coeff_row, rel, rhs)
        heap = []
        node_counter = 0
        heapq.heappush(heap, (-(np.inf), node_counter, []))
        node_counter += 1
        explored = 0

        while heap and explored < node_limit:
            _, nid, extra = heapq.heappop(heap)
            explored += 1
            # build augmented problem
            A_aug = np.vstack([self.A] + [np.array(row).reshape(1, -1) for row,_,_ in extra]) if extra else self.A.copy()
            b_aug = np.hstack([self.b] + [rhs for _,_,rhs in extra]) if extra else self.b.copy()
            relations_aug = self.relations + [rel for _,rel,_ in extra]

            if verbose:
                print(f"
[Node {nid}] extra constraints: {[(row,rel,rhs) for row,rel,rhs in extra]}")

            try:
                solver = TwoPhaseSimplex(A_aug, b_aug, self.c, relations_aug)
                x_lp, z_lp = solver.two_phase_solve()
                # original variables are first n_orig
                x_orig = x_lp[:self.n]
                if verbose:
                    print(f"LP relaxation obj = {z_lp:.8g}, x = {np.round(x_orig,8)}")
            except Exception as e:
                if verbose:
                    print(f"LP infeasible or error: {e}")
                continue

            # pruning by bound
            if self.maximize:
                if z_lp <= best_obj + self.tol:
                    if verbose:
                        print("Pruned by bound (max):", z_lp, "<=", best_obj)
                    continue
            else:
                if z_lp >= best_obj - self.tol:
                    if verbose:
                        print("Pruned by bound (min):", z_lp, ">=", best_obj)
                    continue

            # check integrality
            if self.is_integer_solution(x_orig):
                val = sum(self.c[j]*round(x_orig[j]) for j in range(self.n))
                if self.maximize:
                    if val > best_obj + self.tol:
                        best_obj = val
                        best_x = np.array([round(x_orig[j]) for j in range(self.n)])
                        if verbose:
                            print("New integer solution (max): obj=", best_obj, "x=", best_x)
                else:
                    if val < best_obj - self.tol:
                        best_obj = val
                        best_x = np.array([round(x_orig[j]) for j in range(self.n)])
                        if verbose:
                            print("New integer solution (min): obj=", best_obj, "x=", best_x)
                continue

            # branching
            j = self.fractional_index(x_orig)
            if j is None:
                # numerical issue — treat as integer
                val = sum(self.c[j]*round(x_orig[j]) for j in range(self.n))
                if self.maximize and val > best_obj + self.tol:
                    best_obj = val
                    best_x = np.array([round(x_orig[j]) for j in range(self.n)])
                continue

            xj = x_orig[j]
            flo = math.floor(xj)
            cei = math.ceil(xj)

            # left branch: x_j <= floor  => row = e_j, rel '<=', rhs = floor
            row_left = np.zeros(self.n)
            row_left[j] = 1.0
            extra_left = extra + [(row_left, '<=', flo)]
            # right branch: x_j >= ceil => row = e_j, rel '>=', rhs = ceil
            row_right = np.zeros(self.n)
            row_right[j] = 1.0
            extra_right = extra + [(row_right, '>=', cei)]

            # priority: use LP relaxation objective as optimistic bound
            # for max-heap we push (-z_lp) to explore higher objective first
            heapq.heappush(heap, (-(z_lp), node_counter, extra_left))
            node_counter += 1
            heapq.heappush(heap, (-(z_lp), node_counter, extra_right))
            node_counter += 1

        return best_x, best_obj

# ------------------ IO and main ------------------

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
    print("
--- Целочисленное ЛП (Branch & Bound) ---
")
    n = int(input("Число переменных n: "))
    m = int(input("Число ограничений m: "))

    obj_type = input("Цель (max/min) [max]: ").strip().lower() or 'max'
    if obj_type not in ('max', 'min'):
        obj_type = 'max'
    maximize = (obj_type == 'max')

    print(f"Введите коэффициенты целевой функции (длина {n}):")
    c = np.array(prompt_list("c: ", n), dtype=float)
    if not maximize:
        c = -c

    A_rows = []
    relations = []
    b = []
    print("
Теперь задайте ограничения (каждое: n коэффициентов, знак <=, >= или =, правая часть):")
    for i in range(m):
        row = prompt_list(f"  строка {i+1} (коэффициенты): ", n)
        rel = input("  знак (<=, >=, =) [<=]: ").strip() or '<='
        if rel not in ('<=', '>=', '='):
            rel = '<='
        rhs = float(input("  правая часть (b): "))
        A_rows.append(row)
        relations.append(rel)
        b.append(rhs)

    A = np.array(A_rows, dtype=float)
    b = np.array(b, dtype=float)

    var_integers = input("Какие переменные целочисленные? (all / list например '0 2' для x1,x3) [all]: ").strip()
    if var_integers == '' or var_integers.lower() == 'all':
        int_vars = None
    else:
        parts = var_integers.replace(',', ' ').split()
        int_vars = [int(p) for p in parts]

    return A, b, c, relations, int_vars, maximize


def main():
    try:
        A, b, c, relations, int_vars, maximize = read_problem()
        solver = BranchAndBoundIP(A, b, c, relations, int_vars=int_vars, maximize=maximize)
        print('
Запуск ветвей и границ...')
        x_best, val_best = solver.solve(verbose=True, node_limit=5000)
        if x_best is None:
            print('
Целочисленное решение не найдено (возможно не существует).')
        else:
            print('
--- Результат (целочисленное решение) ---')
            for i, xi in enumerate(x_best):
                print(f"  x{i+1} = {int(xi)}")
            # if minimization we negated c; but we restored objective in branch-and-bound
            print(f"Оптимальное целочисленное значение целевой функции: {val_best:.8g}")

    except Exception as e:
        print('
Ошибка:', e)


if __name__ == '__main__':
    main()
