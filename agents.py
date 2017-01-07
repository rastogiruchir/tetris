from abc import abstractmethod, abstractproperty
from settings import PIECE_GENERATOR_AGENT_INDEX, PLAYER_AGENT_INDEX
import random


class Agent:
    def __init__(self, index=0):
        self.index = index

    @abstractmethod
    def get_action(self, state):
        pass


class RandomPlayerAgent(Agent):
    def get_action(self, state):
        actions = state.get_legal_actions(self.index)
        return random.choice(actions) if len(actions) != 0 else None


def evaluationFunction(board, lines_cleared):
    score = -169.513 * board.findVerticalGaps()
    score += -178.94 * board.findHorizontalGaps()
    score += -61.88 * board.findDensity()
    score += 16.892 * board.findHeightRange()
    score += 12.99 * board.findHeightChanges()
    score += -528.5 * board.findMaxHeight() 
    score += -13 * lines_cleared
    return score

def forwardSearch(board, lines_cleared):
    score = 0
    score += lines_cleared * -149.2932919
    score += board.findVerticalGaps() * -3366.904174
    score += board.findBlockWeight() * -10895.92952
    return score


class ExpectimaxAgent(Agent):
    def __init__(self, index, K, depth):
        self.index = index
        self.K = K
        self.depth = depth

    def lookahead(self, state, action, depth):
        if depth == 0:
            new_state = state.generate_successor(action, PIECE_GENERATOR_AGENT_INDEX)
            actions = [evaluationFunction(action.board, action.lines_cleared) for action in new_state.get_legal_actions(PLAYER_AGENT_INDEX)]
            return max(actions) if len(actions) != 0 else -10000

        new_state = state.generate_successor(action, PLAYER_AGENT_INDEX)
        new_actions = new_state.get_legal_actions(PIECE_GENERATOR_AGENT_INDEX)
        
        value = 0
        for action in new_actions:
            value += self.lookahead(new_state, action, depth - 1)/len(new_actions)

        return value

    def get_action(self, state):
        actions = [(evaluationFunction(action.board, action.lines_cleared), action) for action in state.get_legal_actions(self.index)]
        
        if len(actions) == 0:
            return None

        actions = sorted(actions, key=lambda x: x[0], reverse=True)
        good_actions = actions[0 : self.K] if len(actions) > self.K else actions

        good_actions = [(self.lookahead(state, action, self.depth - 1), action) for (_, action) in good_actions]
        return max(good_actions, key=lambda x: x[0])[1]


class SmartPlayerAgent(Agent):
    def get_action(self, state):
        actions = [(evaluationFunction(action.board, action.lines_cleared), action) for action in state.get_legal_actions(self.index)]
        if len(actions) == 0:
            return None
        return max(actions, key=lambda x: x[0])[1]



class RandomPieceGenerator(Agent):
    def get_action(self, state):
        return random.choice(state.get_legal_actions(self.index))
