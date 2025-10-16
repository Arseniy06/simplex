#!/usr/bin/env python3
"""
Ещё более упрощённая реализация простого симплекс-метода для строго канонической задачи:
  max c^T x
  subject to A x = b, x >= 0

В этой версии убрана явная переменная theta — её вычисление теперь "спрятано"
в части, где выбирается выходящая строка (leave_row). Остальная логика сохранена.

Внимание: при работе с вещественными коэффициентами это может привести к проблемам из‑за плавающей точности,
но для «чётких» входных данных (целые коэффициенты, явные единичные столбцы для базиса) программа работает.

Ввод: n, m, вектор c (длина n), матрица A (m строк по n чисел), вектор b (длина m).
Вывод: пошаговые таблицы и оптимальное решение (значения xi и значение целевой функции).
"""

import sys
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


def read_canonical():
    print("
--- Каноническая задача (max c^T x, Ax = b, x >= 0) ---
")
    n = int(input("Число переменных n: "))
    m = int(input("Число ограничений m: "))

    print(f"Введите коэффициенты цели (длина {n}), пример: 3 1 4")
    c = np.array(prompt_list("c: ", n), dtype=float)

    print(f"
Введите матрицу A построчно ({m} строк, каждая по {n} чисел):")
    A_rows = []
    for i in range(m):
        row = prompt_list(f"  строка {i+1}: ", n)
        A_rows.append(row)
    A = np.array(A_rows, dtype=float)

    print(f"
Введите правую часть b (длина {m}):")
    b = np.array(prompt_list("b: ", m), dtype=float)

    return A, b, c


class SimplexCanonical:
    def __init__(self, A, b, c):
        self.A = A.astype(float)
        self.b = b.astype(float)
        self.c = c.astype(float)
        self.m, self.n = self.A.shape
        self.var_names = [f"x{i+1}" for i in range(self.n)]

        # ensure b >= 0 by flipping rows with negative RHS (strict check)
        for i in range(self.m):
            if self.b[i] < 0:
                self.A[i, :] *= -1.0
                self.b[i] *= -1.0

        # find initial basis: columns that are exact unit vectors (strict equality)
        self.basis = [-1] * self.m
        for j in range(self.n):
            col = self.A[:, j]
            nz = np.where(col != 0)[0]
            if len(nz) == 1 and col[nz[0]] == 1.0:
                row = int(nz[0])
                if self.basis[row] == -1:
                    self.basis[row] = j

        if any(bi == -1 for bi in self.basis):
            missing = [i for i,bi in enumerate(self.basis) if bi == -1]
            raise ValueError(f"Не найден единичный столбец для строк {missing}.
" \
                             "В канонической форме ожидается, что в A присутствует базис (единичные столбцы).")

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

        # objective row: reduced costs (strict arithmetic)
        B = A[:, basis]
        B_inv = np.linalg.inv(B)
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

    def solve(self, max_iters=500, show_iters=True):
        A = self.A.copy()
        b = self.b.copy()
        c = self.c.copy()
        basis = self.basis.copy()
        m, n = A.shape

        for it in range(1, max_iters+1):
            # compute B^{-1}, basic solution
            B = A[:, basis]
            B_inv = np.linalg.inv(B)

            xb = B_inv.dot(b)
            cb = c[basis]
            pi = cb.dot(B_inv)
            reduced = c - pi.dot(A)

            if show_iters:
                self.pretty_tableau(A, b, c, basis, header=f"Итерация {it}")

            # find entering variable (strict > 0)
            entering = None
            for j in range(n):
                if reduced[j] > 0:
                    entering = j
                    break

            if entering is None:
                # optimal
                x = np.zeros(n)
                for i in range(m):
                    x[basis[i]] = xb[i]
                z = float(c.dot(x))
                return x, z, basis

            # compute direction
            a_j = A[:, entering]
            d = B_inv.dot(a_j)

            # choose leaving row without explicit theta variable
            positives = ((xb[i] / d[i], i) for i in range(m) if d[i] > 0)
            try:
                _, leave_row = min(positives, key=lambda t: (t[0], t[1]))
            except ValueError:
                # no positive d[i] found -> unbounded
                raise RuntimeError("Решение неограниченно вверх (unbounded).")

            # pivot
            basis[leave_row] = entering

        raise RuntimeError("Превышено число итераций")


def main():
    try:
        A, b, c = read_canonical()
        solver = SimplexCanonical(A, b, c)
        print('
Начальный базис:', [f"x{j+1}" for j in solver.basis])
        x, z, basis = solver.solve(show_iters=True)

        print('
--- Результат ---')
        for i in range(solver.n):
            print(f"  {solver.var_names[i]:>4} = {x[i]:.8g}")
        print(f"
Оптимальное значение целевой функции: {z:.8g}")

    except Exception as e:
        print('
Ошибка:', e)


if __name__ == '__main__':
    main()
