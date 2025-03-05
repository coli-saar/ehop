# The Graph Coloring Problem

## Problem Description

The Graph Coloring Problem (GCP) is a classic optimization problem in which the goal is to assign colors to the vertices of a graph such that no two adjacent vertices share the same color while minimizing the number of colors used.

[Wikipedia Page](https://en.wikipedia.org/wiki/Graph_coloring)

## Inversion Mechanism

We invert a GCP instance by changing the constraints: the coloring must now not assign two nodes the same color if they are *not* adjacent. To keep the inverted instance equivalent to the original, we present the complement of the original graph.

## Instance Representation

Instances are represented as graphs using the `networkx` package. Each node is given an integer label, with the labels ranging from __1__ to __n__ for a graph with __n__ nodes.
