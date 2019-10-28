from sat import run_solver
from sudoku.sudoku import *
from solvers import iterative_sat


def solve_sudoku(sudoku):
    gen_sudoku_sat(sudoku, fn="sudoku_tmp.in")
    f_in = open("sudoku_tmp.in", "r")
    f_out = open("sudoku_res.out", "w+")
    run_solver(
        f_in,
        f_out,
        iterative_sat,
        brief=False,
        verbose=False,
        output_all=False,
        starting_with="")
    f_in.close()
    f_out.close()

    f_out = open("sudoku_res.out", "r")
    solution = f_out.readline()[:-1]

    board = parse_solution(solution)

    if check_sudoku(board):
        print('grid valid')
    else:
        print('grid invalid')

    return board


board = generate_sudoku(mask_rate=0.4)
draw_sudoku(board)
solved = solve_sudoku(board)
draw_sudoku(solved)
