from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy

from board import Board
from pieces import Piece, PieceNumToPieceClass

points_per_line = 1
points_per_piece = 0.01

class MDP:
    @abstractmethod
    def start_state(self):
        pass

    @abstractmethod
    def actions(self, state):
        pass

    @abstractmethod
    def succ_and_prob_reward(self, state, action):
        pass

    @abstractmethod
    def discount(self):
        pass

class TetrisMDP(MDP):
    def start_state(self):
        return TetrisState(piece=Piece.get_random_piece_set(1)[0])
    def actions(self, state):
        actions = []
        for piece_rot in state.piece.rotations():  # For all rotated versions of the pieces, get all the possible placements
            actions += state.board.get_placements(piece_rot)
        return actions
    def succAndProbReward(self, state, action):
        if action is None:
            return []
        board, lines_cleared = action
        triples = []
        for piece in PieceNumToPieceClass:
            new_state = TetrisState(board=board, piece=piece, lines_cleared=state.lines_cleared + lines_cleared,
                                    score=state.score + lines_cleared, pieces_placed=state.pieces_placed + 1)
            triples.append((new_state, 1.0/len(PieceNumToPieceClass), lines_cleared + lines_cleared**2))
        return triples
    def discount(self):
        return 1

class TetrisState:
    def __init__(self, board=None, piece=None, lines_cleared=0, score=0, pieces_placed=0):
        self.board = board if board is not None else Board()
        self.piece = piece if piece is not None else Piece()
        self.lines_cleared = lines_cleared
        self.score = score
        self.pieces_placed = pieces_placed