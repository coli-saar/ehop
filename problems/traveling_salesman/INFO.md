# The Traveling Salesman Problem

## Problem Description

The Traveling Salesman Problem (TSP) is a classic optimization problem in which a salesman must visit a set of cities exactly once and return to the starting city. The goal is to find the shortest possible route that visits each city exactly once and returns to the starting city. We use the symmetric TSP, where the distance between two cities is the same in both directions.

[Wikipedia Page](https://en.wikipedia.org/wiki/Travelling_salesman_problem)

## Inversion Mechanism

We invert a TSP instance by making the goal to find the *longest* route instead of the shortest one. In order to make the inverted instance equivalent to the original, we shift the edge weights (distances). Shifting is described in our paper and implemented in the `invert_tsp_graph` function (defined in `problems/traveling_salesman/model.py`).

TSP also has an extra inversion mechanism which simply changes the goal to maximizing total distance without shifting the distances. This is called the `"maximize"` variant in the code.

## Instance Representation

Representations were based on the formatting documented by [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf).
