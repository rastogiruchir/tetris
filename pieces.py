import random

import numpy as np
from util import cached_property, stringify_grid


class MetaPiece(type):
    REGISTRY = {}
    def __new__(cls, name, bases, attrs):
        new_cls = super(MetaPiece, cls).__new__(cls, name, bases, attrs)
        if len(bases):
            cls.REGISTRY[name] = new_cls
        return new_cls


class Piece(metaclass=MetaPiece):
    base = np.array((0, 0), dtype='int8')
    piece_type = None

    @classmethod
    def rotations(cls):
        if '_rotations' not in cls.__dict__:
            cls._rotations = [cls(cls.base, 0)]
            num_rots = 1
            rot = np.rot90(cls.base)
            while not np.array_equal(rot, cls.base):
                cls._rotations.append(cls(rot, num_rots))
                rot = np.rot90(rot)
                num_rots += 1
        return cls._rotations

    @cached_property
    def width(self):
        return self.grid.shape[1]

    @cached_property
    def height(self):
        return self.grid.shape[0]

    @cached_property
    def dims(self):
        return self.grid.shape

    @cached_property
    def skirt(self):
        skirt_arr = np.zeros(self.width, dtype='int8')
        for ic in range(self.width):
            for ir in reversed(range(self.height)):
                if self.grid[ir][ic] == 1:
                    skirt_arr[ic] = ir
                    break
        return skirt_arr

    @cached_property
    def eff_height(self):
        eff_height_arr = np.zeros(self.width, dtype='int8')
        for ic in range(self.width):
            for ir in range(self.height):
                if self.grid[ir][ic] == 1:
                    eff_height_arr[ic] = self.skirt[ic] - ir
                    break
        return eff_height_arr

    def __init__(self, grid=None, rotation_index=0):
        self.grid = grid if grid is not None else self.base
        self.rotation_index = rotation_index

    def __str__(self):
        return "{0}".format(stringify_grid(self.grid))

    @classmethod
    def get_random_rotation(cls):
        return random.choice(cls.rotations())

    @classmethod
    def get_random_piece(cls):
        return MetaPiece.REGISTRY[random.choice(list(MetaPiece.REGISTRY.keys()))].get_random_rotation()

    @classmethod
    def get_random_piece_set(cls, num_random):
        piece_set = [cls.get_random_piece() for _ in range(num_random)]
        return piece_set


class IPiece(Piece):
    base = np.array([[1], [1], [1], [1]], dtype='int8')
    piece_type = "I"


class OPiece(Piece):
    base = np.array([[1, 1], [1, 1]], dtype='int8')
    piece_type = "O"


class TPiece(Piece):
    base = np.array([[1, 1, 1], [0, 1, 0]], dtype='int8')
    piece_type = "T"


class SPiece(Piece):
    base = np.array([[0, 1, 1], [1, 1, 0]], dtype='int8')
    piece_type = "S"


class ZPiece(Piece):
    base = np.array([[1, 1, 0], [0, 1, 1]], dtype='int8')
    piece_type = "Z"


class JPiece(Piece):
    base = np.array([[0, 1], [0, 1], [1, 1]], dtype='int8')
    piece_type = "J"


class LPiece(Piece):
    base = np.array([[1, 0], [1, 0], [1, 1]], dtype='int8')
    piece_type = "L"

PieceNumToPieceType = ["I", "O", "T", "S", "Z", "J", "L"]
PieceNumToPieceClass = [IPiece, OPiece, TPiece, SPiece, ZPiece, JPiece, LPiece]
PieceTypeToPieceClass = dict(zip(PieceNumToPieceType, PieceNumToPieceClass))

if __name__ == '__main__':
    for piece in PieceNumToPieceClass:
        p = piece()
        print("Piece {0}".format(p.piece_type))
        print("=======")
        print("Base Grid:")
        print(p)
        print("Dimensions = Height: {0}, Width: {1}".format(*p.dims))
        print("Skirt = {0}".format(p.skirt))
        print("Fringe = {0}".format(p.eff_height))
        print("Rotations ({0}):".format(len(p.rotations())))
        for rr in p.rotations():
            print(rr)
    print("Getting random piece sets...")
    for piece in Piece.get_random_piece_set(10):
        print(piece)
