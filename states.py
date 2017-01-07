from board import Board
from pieces import Piece, PieceNumToPieceType, PieceTypeToPieceClass
from settings import PIECE_GENERATOR_AGENT_INDEX, PLAYER_AGENT_INDEX

from copy import deepcopy


class GameState:
    def __init__(self, board=None, piece=None, lines_cleared=0, curr_combo=0, score=0, pieces_placed=0, lose=False):
        self.board = board if board is not None else Board()
        self.piece = piece if piece is not None else Piece.get_random_piece()
        self.lines_cleared = lines_cleared
        self.curr_combo = curr_combo
        self.score = score
        self.pieces_placed = pieces_placed
        self.lose = lose

    def get_legal_actions(self, agent_index=PLAYER_AGENT_INDEX):
        if self.lose:
            return []
        if agent_index == PLAYER_AGENT_INDEX:
            actions = self.board.get_placements_all_rotations(self.piece)
            if len(actions) == 0:
                self.lose = True
            return actions
        elif agent_index == PIECE_GENERATOR_AGENT_INDEX:
            return PieceNumToPieceType

    def generate_successor(self, action, agent_index=PLAYER_AGENT_INDEX):
        if self.lose:
            return None
        successor_state = deepcopy(self)
        if agent_index == PLAYER_AGENT_INDEX:
            successor_state.board = Board(grid=action.board.grid)
            successor_state.lines_cleared = self.lines_cleared + action.lines_cleared
            successor_state.curr_combo = 0 if action.lines_cleared == 0 else self.curr_combo + 1
            successor_state.score = self.score + scoring_func(action.lines_cleared, successor_state.curr_combo)
            successor_state.pieces_placed = self.pieces_placed + 1
        elif agent_index == PIECE_GENERATOR_AGENT_INDEX:
            successor_state.piece = PieceTypeToPieceClass[action]()
        return successor_state


def scoring_func(lines_cleared, curr_combo):
    return 10 * (0.5 * (curr_combo**2 + lines_cleared**2))
