import random
import argparse
from collections import namedtuple

import numpy as np

from util import cached_property, stringify_grid, get_verbose_print_func
from settings import DEFAULT_BOARD_DIMENSIONS
import pieces

test_mode = False
vprint = get_verbose_print_func(test_mode)

class Board:
    PlacementTuple = namedtuple('PlacementTuple', ['board', 'lines_cleared'])

    @cached_property
    def fringe(self):
        fringe_arr = np.ones(self.width, dtype='int8') * self.height
        for ic in range(self.width):
            for ir in range(self.height):
                if self.grid[ir, ic] != 0:
                    fringe_arr[ic] = ir
                    break
        return fringe_arr

    @cached_property
    def border(self):
        border_arr = self.height - self.fringe
        return border_arr

    @cached_property
    def true_grid(self):
        return np.asarray(list(reversed(self.grid)))

    def findMaxHeight(self):
        return max(self.border)

    def findHeightRange(self):
        return max(self.border) - min(self.border)

    def findHeightChanges(self):
        totalChange = 0
        for column in range(1, self.width):
            delta = self.border[column] - self.border[column - 1]
            totalChange += abs(delta)
        return totalChange
   
    def findBlockWeight(self):
        max_height = max(self.border)
        weight = 0
        for row in range(max_height):
            for col in range(self.width):
                if self.true_grid[row, col] == 1:
                    weight += row
        return weight/100

    def findDensity(self):
        max_height = max(self.border)
        numFilled = 0
        for i in range(max_height):
            numFilled += sum(self.true_grid[i])
        return float(numFilled)/(max_height * self.width) if max_height > 0 else 1

    def findHorizontalGaps(self):
        max_height = max(self.border)
        gaps = 0
        for row in range(max_height):
            for col in range(1, self.width):
                if self.true_grid[row, col] != self.true_grid[row, col - 1]:
                    gaps += 0.1
        return gaps

    def findVerticalGaps(self):
        max_height = max(self.border)
        gaps = 0
        for col in range(self.width):
            for row in range(1, self.border[col]):
                if self.true_grid[row, col] != self.true_grid[row - 1, col]:
                    gaps += 0.1
        return gaps


    def __init__(self, dims=DEFAULT_BOARD_DIMENSIONS, grid=None):
        self.dims = grid.shape if grid is not None else dims
        self.height, self.width = self.dims
        self.grid = grid if grid is not None else np.zeros(dims, dtype='int8')

    def __str__(self):
        return stringify_grid(self.grid)

    def clear_rows(self):
        to_remove = [ir for ir, r in enumerate(self.grid) if np.count_nonzero(r) == self.width]
        if len(to_remove):
            self.grid = np.delete(self.grid, to_remove, axis=0)
            self.grid = np.insert(self.grid, [0] * len(to_remove), [0], axis=0)
            return len(to_remove)
        return 0

    def get_placements(self, piece):
        placements = []
        for ic in range(self.width - piece.width + 1):
            placement = None
            for iic in range(piece.width):
                modded_skirt = piece.skirt - piece.skirt[iic] * np.ones(piece.width)
                vprint([(self.fringe[ic + iic] - 1 + ms, self.fringe[ic + iic] - 1 + ms - e)
                       for ms, f, e in zip(modded_skirt, self.fringe[ic:ic + piece.width], piece.eff_height)])
                # if all([(0 <= self.fringe[ic + iic] - 1 + ms < f) and (f - 1 - ms >= 0)
                #         for ms, f, s in zip(modded_skirt, self.fringe[ic:ic + piece.width], piece.skirt)]):
                if all([(self.fringe[ic + iic] - 1 + ms < f) and (self.fringe[ic + iic] - 1 + ms - e >= 0)
                       for ms, f, e in zip(modded_skirt, self.fringe[ic:ic + piece.width], piece.eff_height)]):
                    placement = iic
                    break
            if placement is None:
                vprint("skip")
                continue
            else:
                top_row = self.fringe[ic + placement] - piece.skirt[placement] - 1
                grid = np.copy(self.grid)
                for r in range(top_row, top_row + piece.height):
                    for c in range(ic, ic + piece.width):
                        grid[r, c] += piece.grid[r - top_row, c - ic]
                new_board = Board(grid=grid)
                lines_cleared = new_board.clear_rows()
                placements.append(self.PlacementTuple(new_board, lines_cleared))
                vprint(new_board)
        return placements

    def get_placements_all_rotations(self, piece):
        placements = []
        for piece_rot in piece.rotations():
            placements += self.get_placements(piece_rot)
        return placements

class BoardTest:
    AssertionTuple = namedtuple('AssertionTuple', ['func', 'name', 'msg'])
    assertions = []

    @staticmethod
    def register_assertion(assert_fn, assert_name=None, assert_message=''):
        assert_name = assert_name if assert_name is not None else str(len(BoardTest.assertions) + 1)
        BoardTest.assertions.append(BoardTest.AssertionTuple(assert_fn, assert_name, assert_message))

    @property
    def piece_data(self):
        return [(p.piece_type, p.rotation_index) for p in self.piece_list]

    def __init__(self, num_pieces=10, dims=DEFAULT_BOARD_DIMENSIONS,
                 piece_list=None, chosen_boards=None, board=None):
        if piece_list is not None:
            self.piece_list = [pieces.PieceTypeToPieceClass[t].rotations()[r] for t, r in piece_list]
        else:
            self.piece_list = pieces.Piece.get_random_piece_set(num_pieces)
        self.chosen_boards = chosen_boards if chosen_boards is not None else []
        self.board = board if board is not None else Board(dims=dims)
        self.pieces_placed = 0
        self.dims = dims
        self.curr_piece = None

    def print_pieces(self):
        for p in self.piece_list:
            print(p)

    def step(self):
        if self.pieces_placed >= len(self.piece_list):
            print("All pieces placed")
            return False
        self.curr_piece = self.piece_list[self.pieces_placed]
        vprint("Discovering all placements for piece:\n{0}".format(self.curr_piece))
        placements = self.board.get_placements(self.curr_piece)
        if not len(placements):
            print("Piece cannot be placed")
            return False
        for assertion in self.assertions:
            for placement in placements:
                try:
                    assert assertion.func(placement.board) == True
                except AssertionError as e:
                    print("Failed assertion: {0}. {1}. Failed case = \n {1}".format(assertion.name, assertion.msg,
                                                                                    placement.board))
                    raise e
            vprint("Passed assertion: {0}".format(assertion.name))
        vprint("Piece placement summary:")
        vprint("Tried to place piece:\n{0}".format(self.curr_piece))
        vprint("Found {0} valid placements".format(len(placements)))
        vprint("Found {0} invalid placements".format(self.board.width - self.curr_piece.width - len(placements) + 1))
        if self.pieces_placed >= len(self.chosen_boards):
            self.chosen_boards.append(random.randint(0, len(placements) - 1))
        self.board = placements[self.chosen_boards[self.pieces_placed]].board
        vprint("Randomly chosen board:\n{0}".format(self.board))
        self.pieces_placed += 1
        return True

    def run(self, interactive=False):
        for _ in self.piece_list:
            if not self.step(): break
            if interactive:
                input("Press Enter to continue.")

    def print_test_data(self):
        print("BoardTest(chosen_boards={0}, piece_list={1}, dims={2})".format(
            self.chosen_boards, self.piece_data, self.dims))

    @staticmethod
    def run_many_tests(num_tests, *args, **kwargs):
        for t in range(num_tests):
            board_test = BoardTest(*args, **kwargs)
            print("Running test #{0}".format(t + 1))
            try:
                board_test.run()
            except AssertionError as e:
                board_test.print_test_data()
                raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Board Implementation for Tetris.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display all debug and test data')
    args = parser.parse_args()

    BoardTest.register_assertion(lambda b: (b.grid > 1).sum() == 0, 'Overlap', 'Cell overridden with value > 1')
    BoardTest.register_assertion(lambda b: (b.fringe > b.height).sum() == 0, 'Fringe', 'Fringe above the Board height')

    vprint = get_verbose_print_func(args.verbose)
    test = BoardTest(dims=(5, 10))


