#%matplotlib inline
import matplotlib.pyplot as plt
import random
import heapq
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations

class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __repr__(self): return '<{}>'.format(self.state)
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost


failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.


def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]



FIFOQueue = deque

LIFOQueue = list

class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = [] # a heap of (score, item) pairs
        for item in items:
            self.add(item)

    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]

    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)


def breadth_first_search(problem):
    "Search shallowest nodes in the search tree first."
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        return node
    frontier = FIFOQueue([node])
    reached = {problem.initial}
    #print(node)
    while frontier:
        node = frontier.pop()
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                return child
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)
    return failure


def is_cycle(node, k=30):
    "Does this node form a cycle of length k or less?"
    def find_cycle(ancestor, k):
        return (ancestor is not None and k > 0 and
                (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1)))
    return find_cycle(node.parent, k)

def depth_first_recursive_search(problem, node=None):
    if node is None:
        node = Node(problem.initial)
    if problem.is_goal(node.state):
        return node
    elif is_cycle(node):
        return failure
    else:
        for child in expand(problem, node):
            result = depth_first_recursive_search(problem, child)
            if result:
                return result
        return failure

def best_first_search(problem, f):
    "Search nodes with minimum f(node) value first."
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure

def g(n): return n.path_cost

def astar_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + h(n))


def solveProblemAStar_h1(problem):
    """
    Recibe una instancia de SnakeProblem y retorna una lista con la secuencia de acciones que resuelve el problema.
    La solución debe ser óptima (mínimo número de pasos).
    """
    sol = astar_search(problem)

    actions = path_actions(sol) # A list with the actions that solve the problem
    states = path_states(sol) # A list with the intermediate states that result of applying the list of actions

    solucion = []

    for i in actions:
      solucion.append(i[0])

    return solucion

def solveProblemAStar_h2(problem):
    """
    Recibe una instancia de SnakeProblem y retorna una lista con la secuencia de acciones que resuelve el problema.
    La solución debe ser óptima (mínimo número de pasos).
    """
    sol = astar_search(problem)

    actions = path_actions(sol) # A list with the actions that solve the problem
    states = path_states(sol) # A list with the intermediate states that result of applying the list of actions

    solucion = []

    for i in actions:
      solucion.append(i[0])

    return solucion



import copy
import numpy as np
from snakeai.gameplay.entities import SnakeAction, Snake, Point, CellType

class SnakeProblem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    direcciones = [SnakeAction.TURN_LEFT,
                   SnakeAction.MAINTAIN_DIRECTION,
                   SnakeAction.TURN_RIGHT]

    fruit = None
    heuristica = None

    def __init__(self, initial=None, **kwds):
        if 'heuristica' in kwds : self.heuristica = kwds['heuristica']
        self.__dict__.update(initial=initial, **kwds)

    def get_observation(self, state):
        """ Observe the state of the environment. """
        if state.is_game_over:
            return (0, 0, 0)
        center = state.snake.head + state.snake.direction
        if state.snake.direction == Point(0,1):
            left = state.snake.head + Point(1,0)
            right = state.snake.head + Point(-1, 0)
        elif state.snake.direction == Point(0, -1):
            left = state.snake.head + Point(-1, 0)
            right = state.snake.head + Point(1, 0)
        elif state.snake.direction == Point(1, 0):
            left = state.snake.head + Point(0, -1)
            right = state.snake.head + Point(0, 1)
        else:
            left = state.snake.head + Point(0, 1)
            right = state.snake.head + Point(0, -1)
        return (state.field[left], state.field[center], state.field[right])


    def actions(self, state):

        #if state.timestep_index == 0:
        #  print("Tablero inicial")
        #  print(state.field)

        self.fruit = state.fruit
        stateTuples = []
        observation = self.get_observation(state)

        for i in range(len(observation)):
          stateCopy = copy.copy(state)
          stateCopy.field = copy.deepcopy(state.field)
          stateCopy.snake = copy.deepcopy(state.snake)
          if observation[i] == 0 or observation[i] == 1:
            stateTuples.append(tuple([self.direcciones[i], stateCopy ]))

        return stateTuples

    def result(self, state, action):
        act, state1 = action

        if act == 0:
          state1.timestep()
        elif act == 1:
          state1.snake.turn_left()
          state1.timestep()
        elif act == 2:
          state1.snake.turn_right()
          state1.timestep()
        return state1

    def is_goal(self, state):
        return True if state.is_game_over or self.fruit == state.snake.head else False

    def goal_test(self, state):
        return False

    def action_cost(self, s, a, s1):
        return 1

    def h(self, node):
        if self.heuristica == None:
          return 0
        elif self.heuristica == "Euclidiana":
          d = math.sqrt((node.state.fruit.x - node.state.snake.head.x)**2 + (node.state.fruit.y - node.state.snake.head.x)**2)
          return d
        elif self.heuristica == "Manhattan":
          p = Point(node.state.snake.head.x, node.state.fruit.y)
          c1 = math.sqrt((node.state.snake.head.x - p.x)**2 + (node.state.snake.head.y - p.y)**2)
          c2 = math.sqrt((node.state.fruit.x - p.x)**2 + (node.state.fruit.y - p.y)**2)
          return c1 + c2

        return 0
    def __str__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self.initial, self.goal)
