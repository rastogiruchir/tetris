import mdp as mdp
import board as board
import random
import util
import collections
import math
from numpy.linalg import norm


class RLAlgorithm():
    def getAction(self, state):
        raise Exception('Override me!')

    def incorporateFeedback(self, state, action, reward, newState):
        raise Exception('Override me!')

class QLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.defaultdict(float)
        self.numIters = 0

    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    def getV(self, state):
        if state is None:
            return None
        
        Q_values = []
        for action in self.actions(state):
            Q_values.append(self.getQ(state, action))

        return None if len(Q_values) == 0 else max(Q_values)

    def getAction(self, state):
        self.numIters += 1   

        if len(self.actions(state)) == 0:
            return None

        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            actions = [(self.getQ(state, action), action) for action in self.actions(state)]
            actions = sorted(actions, reverse=True, key= lambda x: x[0])
            bestActions = [action for action in actions if action[0] == actions[0][0]]
            return random.choice(bestActions)[1]

    def getStepSize(self):
        return 1.0 / self.numIters

    def incorporateFeedback(self, state, action, reward, newState):
        featureVector = self.featureExtractor(state, action)
        V_opt = self.getV(newState)

        if V_opt is not None:
            gradCoeff = self.getQ(state, action) - (reward + self.discount * V_opt)
        else:
            gradCoeff = -1 * reward #+ norm(list(self.weights.values()), 2)

        gradient = {key: value * gradCoeff for (key, value) in featureVector}

        for key, value in gradient.items():
            self.weights[key] = self.weights.get(key, 0) - value * self.getStepSize()

###################################################################################################

def simulate(mdp, rl, numTrials=100, maxIterations=100, verbose=False):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.start_state()
        sequence = [state]
        totalReward = 0
        totalDiscount = 1
        for t in range(maxIterations):
            if verbose == True:
                print(util.stringify_grid(state.board.grid))
            action = rl.getAction(state)
            transitions = mdp.succAndProbReward(state, action)
            if len(transitions) == 0:
                reward = -100
                rl.incorporateFeedback(state, action, reward, None)
                totalReward += totalDiscount * reward
                break
    
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence += [action, reward, newState]
            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount()
            state = newState

        print ("Trial %d (totalReward = %s)" % (trial, totalReward))
        print (rl.weights)
        totalRewards.append(totalReward)
    return totalRewards


def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


def featureExtractorHelper(board, lines_cleared):
    features = []
    features.append(("lines_cleared", lines_cleared))
    features.append(("height_range", board.findHeightRange()))
    features.append(("height_changes", board.findHeightChanges()))
    features.append(("density", board.findDensity()))
    features.append(("vertical_gaps", board.findVerticalGaps()))
    features.append(("horizontal_gaps", board.findHorizontalGaps()))
    features.append(("max_height", board.findMaxHeight()))
    features.append(("block_weights", board.findBlockWeight()))
    return features

def improvedFeatureExtractor(state, action):
    features = []
    if action is not None:
        return featureExtractorHelper(action.board, action.lines_cleared)
    else:
        return featureExtractorHelper(state.board, 0)


def main():
    MDP = mdp.TetrisMDP()
    rl = QLearningAlgorithm(MDP.actions, 1, improvedFeatureExtractor)
    simulate(MDP, rl, numTrials=50, verbose=False)


if __name__ == '__main__':
    main()
