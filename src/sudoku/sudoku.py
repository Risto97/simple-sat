import numpy as np


# https://github.com/MorvanZhou/sudoku
def generate_sudoku(mask_rate=0.5):
    while True:
        n = 9
        m = np.zeros((n, n), np.int)
        rg = np.arange(1, n + 1)
        m[0, :] = np.random.choice(rg, n, replace=False)
        try:
            for r in range(1, n):
                for c in range(n):
                    col_rest = np.setdiff1d(rg, m[:r, c])
                    row_rest = np.setdiff1d(rg, m[r, :c])
                    avb1 = np.intersect1d(col_rest, row_rest)
                    sub_r, sub_c = r // 3, c // 3
                    avb2 = np.setdiff1d(
                        np.arange(0, n + 1),
                        m[sub_r * 3:(sub_r + 1) * 3, sub_c * 3:(sub_c + 1) *
                          3].ravel())
                    avb = np.intersect1d(avb1, avb2)
                    m[r, c] = np.random.choice(avb, size=1)
            break
        except ValueError:
            pass
    mm = m.copy()
    mm[np.random.choice([True, False],
                        size=m.shape,
                        p=[mask_rate, 1 - mask_rate])] = 0
    return mm.tolist()


def parse_solution(solution):
    solution = solution.split(" ")
    board = np.zeros([9, 9], dtype=np.int8)
    for literal in solution:
        if "~" not in literal:
            p = literal.find('p')
            col, row = literal[p + 1:-1]
            val = int(literal[-1])
            board[int(col) - 1][int(row) - 1] = val
    return board


# https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
def draw_sudoku(board):
    base = 3  # Will generate any size of random sudoku board in O(n^2) time
    side = base * base

    # for line in board: print(line)
    def expandLine(line):
        return line[0] + line[5:9].join(
            [line[1:5] * (base - 1)] * base) + line[9:13]

    line0 = expandLine("╔═══╤═══╦═══╗")
    line1 = expandLine("║ . │ . ║ . ║")
    line2 = expandLine("╟───┼───╫───╢")
    line3 = expandLine("╠═══╪═══╬═══╣")
    line4 = expandLine("╚═══╧═══╩═══╝")

    symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums = [[""] + [symbol[n] for n in row] for row in board]
    print(line0)
    for r in range(1, side + 1):
        print("".join(n + s for n, s in zip(nums[r - 1], line1.split("."))))
        print([line2, line3, line4][(r % side == 0) + (r % base == 0)])


# https://scipython.com/book/chapter-6-numpy/examples/checking-a-sudoku-grid-for-validity/
def check_sudoku(grid):
    """ Return True if grid is a valid Sudoku square, otherwise False. """
    for i in range(9):
        # j, k index top left hand corner of each 3x3 tile
        j, k = (i // 3) * 3, (i % 3) * 3
        if len(set(grid[i,:])) != 9 or len(set(grid[:,i])) != 9\
                   or len(set(grid[j:j+3, k:k+3].ravel())) != 9:
            return False
    return True


def gen_sudoku_sat(sudoku, fn="sudoku_tmp.in"):
    dim = (9, 9)

    f = open(fn, "w+")

    # --------- Pre filled -------------

    for row, line in enumerate(sudoku):
        for col, val in enumerate(line):
            if val != 0:
                print(f"p{row+1}{col+1}{val}", file=f)

    # -------- Individual Cell Clauses -----------
    # at least one number in each cell
    for row in range(1, dim[0] + 1):
        for col in range(1, dim[1] + 1):
            for i in range(1, 10):
                print(f"p{row}{col}{i}", end=' ', file=f)

            print(f"\n", end='', file=f)

    # every cell can contain only one value
    for row in range(1, dim[0] + 1):
        for col in range(1, dim[1] + 1):
            for i in range(1, 10):
                for j in range(i + 1, 10):
                    print(f"~p{row}{col}{i} ~p{row}{col}{j}", file=f)

    # ----------- Row Clauses ---------------------
    # every row contains at least one of every value
    for row in range(1, dim[0] + 1):
        for i in range(1, 10):
            for col in range(1, dim[1] + 1):
                print(f"p{row}{col}{i}", end=' ', file=f)

            print(f"\n", end='', file=f)

    # row does not contain more than one of value
    for i in range(1, 10):
        for row in range(1, dim[0] + 1):
            for col in range(1, dim[1] + 1):
                for sub_col in range(col + 1, dim[1] + 1):
                    print(f"~p{row}{col}{i} ~p{row}{sub_col}{i}", file=f)

    # -------------- Col Clauses -----------------------
    # every col contains at least one of every value
    for col in range(1, dim[1] + 1):
        for i in range(1, 10):
            for row in range(1, dim[0] + 1):
                print(f"p{row}{col}{i}", end=' ', file=f)

            print(f"\n", end='', file=f)

    # col does not contain more than one of value
    for i in range(1, 10):
        for col in range(1, dim[1] + 1):
            for row in range(1, dim[0] + 1):
                for sub_row in range(row + 1, dim[1] + 1):
                    print(f"~p{row}{col}{i} ~p{sub_row}{col}{i}", file=f)

    # ---------- Block Clauses ---------------------------
    for block_row in [0, 1, 2]:
        for block_col in [0, 1, 2]:
            for i in range(1, 10):
                for row in [1, 2, 3]:
                    for col in [1, 2, 3]:
                        print(
                            f"p{row+block_row*3}{col+block_col*3}{i}",
                            end=' ',
                            file=f)

                print(f"\n", end='', file=f)

    f.close()
