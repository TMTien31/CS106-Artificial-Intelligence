# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def minimax(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = minimax(gameState)

        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        # return result        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def maxValue(state, depth, alpha, beta):
            if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            bestValue = None
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = minValue(successor, 1, depth, alpha, beta)
                if bestValue is None or value > bestValue:
                    bestValue = value
                if alpha is None or value > alpha:
                    alpha = value
                if beta is not None and bestValue is not None and bestValue > beta:
                    break
            return bestValue if bestValue is not None else self.evaluationFunction(state)

        def minValue(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            bestValue = None
            numAgents = gameState.getNumAgents()

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == numAgents - 1:
                    value = maxValue(successor, depth + 1, alpha, beta)
                else:
                    value = minValue(successor, agentIndex + 1, depth, alpha, beta)

                if bestValue is None or value < bestValue:
                    bestValue = value
                if beta is None or value < beta:
                    beta = value
                if alpha is not None and bestValue is not None and bestValue < alpha:
                    break
            return bestValue if bestValue is not None else self.evaluationFunction(state)

        bestAction = None
        bestScore = None
        alpha = None
        beta = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minValue(successor, 1, 1, alpha, beta)
            if bestScore is None or score > bestScore:
                bestScore = score
                bestAction = action
            if alpha is None or score > alpha:
                alpha = score

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def maxValue(state, depth):
            if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            bestValue = None
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = expValue(successor, 1, depth)
                if bestValue is None or value > bestValue:
                    bestValue = value
            return bestValue if bestValue is not None else self.evaluationFunction(state)

        def expValue(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            total = 0
            numActions = len(actions)
            numAgents = gameState.getNumAgents()

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == numAgents - 1:
                    value = maxValue(successor, depth + 1)
                else:
                    value = expValue(successor, agentIndex + 1, depth)
                total += value  

            return total / numActions

        bestAction = None
        bestScore = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expValue(successor, 1, 1)
            if bestScore is None or score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    if newCapsules:
        closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    else:
        closestCapsule = 0

    if closestCapsule:
        closest_capsule = -3 / closestCapsule
    else:
        closest_capsule = 100

    if closestGhost:
        ghost_distance = -2 / closestGhost
    else:
        ghost_distance = -500

    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        closestFood = 0

    return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule

def mybetterEvaluationFunction(currentGameState):
    """
    Evaluation(s) = Score(s) + RewardGain - PenaltyRisk - PenaltyIncompletion
    """

    from util import manhattanDistance

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghost.scaredTimer for ghost in ghostStates]

    score = currentGameState.getScore()
    rewardGain = 0

    # reward: Nếu ghost bị dọa và gần, khuyến khích tiếp cận để ăn ghost
    for i, ghost in enumerate(ghostStates):
        dist = manhattanDistance(pos, ghost.getPosition())
        scared = scaredTimers[i]

        if scared > 0:
            if dist <= 2:
                rewardGain += 100
            else:
                rewardGain += 10 / dist

    # reward: Gần food → khuyến khích ăn food
    if foodList:
        minFoodDist = min(manhattanDistance(pos, f) for f in foodList)
        rewardGain += 10.0 / (minFoodDist + 1) 
    else:
        rewardGain += 100 

    # reward: Gần capsule → khuyến khích lấy capsule
    if capsules:
        capsuleDist = [manhattanDistance(pos, c) for c in capsules]
        minCapsuleDist = min(capsuleDist)
        rewardGain += 5.0 / (minCapsuleDist + 1)

    # PenaltyRisk: Phạt nếu có nguy cơ bị ghost ăn
    penaltyRisk = 0
    for i, ghost in enumerate(ghostStates):
        dist = manhattanDistance(pos, ghost.getPosition())
        scared = scaredTimers[i]

        if scared == 0 and dist <= 2: 
            penaltyRisk += 100

    # PenaltyIncompletion: Phạt nếu chưa hoàn thành nhiệm vụ
    penaltyIncompletion = 0
    penaltyIncompletion += 2 * len(foodList)
    penaltyIncompletion += 10 * len(capsules)

    evaluation = score + rewardGain - penaltyRisk - penaltyIncompletion
    return evaluation

# Abbreviation
better = betterEvaluationFunction
#better = mybetterEvaluationFunction