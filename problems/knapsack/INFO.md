# The Knapsack Problem

## Problem Description

The Knapsack Problem (KSP) is an optimization problem in which the goal is to maximize the total value of items placed into a knapsack without exceeding its capacity. Each item has an associated weight and value, and the knapsack has a maximum (weight) capacity. The goal is to select a subset of items to maximize the total value while ensuring that the total weight does not exceed the capacity of the knapsack. We use the 0-1 KSP, where each item can be either included (once) or excluded from the knapsack (rather than being able to take multiple copies of the same item).

[Wikipedia Page](https://en.wikipedia.org/wiki/Knapsack_problem)

## Inversion Mechanism

We invert a KSP instance by making the goal to select a set of items with *minimal* value whose weight is *at least* equal to the capacity of the knapsack. We also change the knapsack's capacity to $(\sum w_i) - C$, so that the inverted instance effectively asks which items should be left out of the knapsack in the original problem.

## Instance Representation

Instances are represented as a list of profits/values and a list of corresponding weights for each item. The knapsack capacity is also provided as an integer value.
