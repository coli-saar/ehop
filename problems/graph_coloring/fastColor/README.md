# fastColor Readme

This solver was downloaded from [here](https://lcs.ios.ac.cn/~caisw/Color.html). There is also [a paper](https://www.ijcai.org/proceedings/2017/0073.pdf) describing its development.

The following was originally provided in the readme file (code example formatted differently to fit markdown standards):

> This is a graph coloring solver for coloring massive graphs within short time. It interleaves between graph reduction and bound computation.
>
> Usage: `./fastColor -f <instance> -t <cutoff time>`

The `fastColor` file is an ELF file, so it can only be run in a Linux environment. The input file is expected to follow DIMACS formatting for graph coloring problems, and it is expected to NOT have edges recorded in both directions. The cutoff time is in seconds.
