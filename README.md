# TSP with Genetic Algorithms
## Introduction

- SFU CMPT 310: Artificial Intelligence Survey, Individual Project: Solving the TSP with Genetic Algorithms.
- This assignment is to implement the genetic algorithm framework for solving the TSP problem.
- Basically, the framework is modified based on the pseudocode written in Figure 4.8 of the Norvig and Russel textbook.


## Libraries

The following packages are needed for this assignment:
- math
- time
- getopt
- random
- sys

## Instructions

The `tsp.py` is implemented with a command line interface, which allows to set the parameters for running the genetic algorithm through the command line. The following are some options it contains: 

-	`-h` (or `--help`): print command line interface help manual.
-	`-i $arg` (or `--city_ifile $arg`): set the input city dataset filename.
-	`-o $arg` (or `--solution_ofile $arg`): set the output filename for writing the best solution. $arg has default value: best_solution.txt.
-	`-s $arg` (or `--init_pop_generator $arg`): set the initial population generator. Valid options include: random, random_2opt, nearest_neighbor_2opt. $arg has a default value: random.
-	`-c $arg` (or `--crossover $arg`): set the crossover generator. Valid options include: pmx, ox. $arg has a default value: pmx.
-	`-m $arg` (or `--mutation $arg`): set the mutation operator. Valid options include: rand_swap, cim, rsm, 2opt. $arg has a default value: rsm.
-	`-n $arg` (or `--max_gen $arg`): set the maximum number of generations for running the genetic algorithm. $arg has a default value: 10000.
-	`-k $arg` (or `--max_gen $arg`): set the population size. $arg has a default value: 20.

## Examples of Execution

- Print the help manual: 
  
    `python3 tsp.py -h`.
- Generate a tsp solution with all the default arguments:
  
    `python3 tsp.py -i cities1000.txt`.
- Specify the number of generations as 100, and population size as 40: 
  
    `python3 tsp.py -i cities1000.txt -n 1000 -k 40`.
- Specify the init_pop_generator as random_2opt, crossover as ox, mutation operator as rsm: 
  
    `python3 tsp.py -i cities1000.txt -s random_2opt -c ox -m rsm`.

## License

This work is licensed under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (or any later version). 

`SPDX-License-Identifier: Apache-2.0-or-later`

## Disclaimer

**This repository is *ONLY* for backup. Students should *NEVER* use this repository to finish their works, *IN ANY WAY*.**

It is expected that within this course, the highest standards of academic integrity will be maintained, in
keeping with SFU’s Policy S10.01, `Code of Academic Integrity and Good Conduct`.

In this class, collaboration is encouraged for in-class exercises and the team components of the assignments, as well
as task preparation for group discussions. However, individual work should be completed by the person
who submits it. Any work that is independent work of the submitter should be clearly cited to make its
source clear. All referenced work in reports and presentations must be appropriately cited, to include
websites, as well as figures and graphs in presentations. If there are any questions whatsoever, feel free
to contact the course instructor about any possible grey areas.

Some examples of unacceptable behaviour:
- Handing in assignments/exercises that are not 100% your own work (in design, implementation,
wording, etc.), without a clear/visible citation of the source.
- Using another student's work as a template or reference for completing your own work.
- Using any unpermitted resources during an exam.
- Looking at, or attempting to look at, another student's answer during an exam.
- Submitting work that has been submitted before, for any course at any institution.

All instances of academic dishonesty will be dealt with severely and according to SFU policy. This means
that Student Services will be notified, and they will record the dishonesty in the student's file. Students
are strongly encouraged to review SFU’s Code of Academic Integrity and Good Conduct (S10.01) available
online at: http://www.sfu.ca/policies/gazette/student/s10-01.html.

## Author

Zeyong, JIN

April 20, 2020
