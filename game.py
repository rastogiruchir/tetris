from states import GameState
from agents import RandomPlayerAgent, RandomPieceGenerator, SmartPlayerAgent, ExpectimaxAgent, ForwardSearchAgent
from settings import PLAYER_AGENT_INDEX, PIECE_GENERATOR_AGENT_INDEX
import sys

class Game:
    def __init__(self, verbose=True):
        self.curr_state = GameState()
        self.agents = [
            RandomPieceGenerator(PIECE_GENERATOR_AGENT_INDEX),
            ExpectimaxAgent(PLAYER_AGENT_INDEX, 3, 2)
        ]
        self.agent_index = 0
        self.verbose = verbose

    def step(self):
        action = self.agents[self.agent_index].get_action(self.curr_state)
        if action is None:
            self.curr_state.lose = True
            print("Score: %0.2f, lines cleared = %d, pieces placed = %d" % (self.curr_state.score, self.curr_state.lines_cleared, self.curr_state.pieces_placed))
            return self.curr_state.score, self.curr_state.lines_cleared, self.curr_state.pieces_placed
        
        self.curr_state = self.curr_state.generate_successor(action, self.agent_index)
        self.agent_index = (self.agent_index + 1) % len(self.agents)
        
        if self.verbose: 
            print(self.curr_state.board)
        
        return self.curr_state.score, self.curr_state.lines_cleared, self.curr_state.pieces_placed

    def run(self):
        while not self.curr_state.lose:
            score, lines_cleared, pieces_placed = self.step()
        return score, lines_cleared, pieces_placed

def main():
    num_games = 1 if len(sys.argv) == 1 else int(sys.argv[1])
    average_score, average_lines_cleared, average_pieces_placed = 0, 0, 0
    for _ in range(num_games):
        game = Game(verbose=False)
        score, lines_cleared, pieces_placed = game.run()
        average_score += score/num_games
        average_lines_cleared += lines_cleared/num_games
        average_pieces_placed += pieces_placed/num_games
    print()
    print("Average score: %0.2f, average lines cleared: %0.2f, average pieces placed: %0.2f" % (average_score, average_lines_cleared, average_pieces_placed))



if __name__ == '__main__':
    main()

