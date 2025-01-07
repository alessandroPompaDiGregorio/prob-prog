This repository contains the experiments for my thesis: A comparative analysis of verification techniques for probabilistic programs.
For more details regarding the tools that I have used, you can visit:
- https://github.com/moves-rwth/cegispro2
- https://github.com/Luisa-unifi/probabilistic_programming/blob/main/TSI.py
## TSI

The value of eps is 0.005. This allows to obtain a delta which is less than 0.001 (often much less then 0.001), with the only exception of *die_conditioning* which produces a delta of approximately 0.015. This exception is justified by the fact that there is an observation in the program witnessing a rare event ($p\simeq 0.083$). Nonetheless, the TSI exp captures the true expected value.
The programs that do not contain a while loop, do not have a maximum number of iterations, this is indicated with a dash in the corresponding cell.

| Program name      | TSI time | TSI exp                              | t (maximum iterations for TSI) | N (tensor dimensions) | Additional details                                                                                                  |
| ----------------- | -------- | ------------------------------------ | ------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| brp_10            | 3.18     | [0.004, 0.014]                       | 100                            | $10^6$                |                                                                                                                     |
| brp               | 0.36     | [-0.005, 0.005]                      | 10                             | $10^6$                |                                                                                                                     |
| brp               | 3.27     | [-0.005, 0.005]                      | 100                            | $10^6$                |                                                                                                                     |
| brp_finite_family | 0.42     | [-0.005, 0.005]                      | 10                             | $10^6$                |                                                                                                                     |
| my_chain_5        | 0.26     | [0.667, 0.677]                       | 10                             | $10^6$                | $p=0.67232$                                                                                                         |
| grid_small        | 0.63     | [-0.003, 0.006]                      | 20                             | $10^6$                | $p=\frac{1}{2^{10}}= 0.0009765625$                                                                                  |
| gambler           | 0.88     | \[0.494, 0.504\]                     | 20                             | $10^6$                | $p=\frac12=0.5$                                                                                                     |
| geo0              | 0.64     | \[4.954, 4.964\]                     | 10                             | $10^6$                | $\mathbb{E}[z]=5$                                                                                                   |
| geo0              | 0.64     | \[3.952, 3.962\]                     | 20                             | $10^6$                | $\mathbb{E}[z]=z+4$; TSI exp represents z_final - z_initial. Initially z is a tensor of random values in $[0,10^6]$ |
| geo0_obs          | 0.65     | [6.906, 6.916]                       | 20                             | $10^6$                | Contains observe. $\mathbb{E}[z]=7$                                                                                 |
| die_conditioning  | 0.04     | [4.997, 5.007]                       | -                              | $10^6$                | Contains observe. $\mathbb{E}[d_1]=5$                                                                               |
| PrinSys           | 0.41     | [0.995, 1.005]                       | 10                             | $10^6$                |                                                                                                                     |
| RevBin            | 0.33     | [11.763, 11.773]                     | 10                             | $10^6$                |                                                                                                                     |
| sprdwalk          | 0.31     | [2.365,2.375]; lower prob = 0.986897 | 10                             | $10^6$                | exp represents expected runtime. Lower prob is term/(term+live)                                                     |
| normal            | 0.02     | [0.527, 0.537]                       | -                              | $10^6$                | $p=0.5342$                                                                                                          |
| Monty Hall        | 0.39     | [0.661, 0.671]                       | -                              | $10^6$                | Contains an observe statement; $p=\frac23$                                                                          |
| Burglar Alarm     | 0.08     | [0.024, 0.034]                       | -                              | $10^6$                | Contains an observe statement                                                                                       |

## cegispro2

In this table we verify the upper bounds given by TSI using cegispro2. TO=5min

| Program name      | cegispro2 total time | Number of CTIs | POSTEXP                                 | PROP                                   | Additional details                         |
| ----------------- | -------------------- | -------------- | --------------------------------------- | -------------------------------------- | ------------------------------------------ |
| brp_10            | 167.71               | 95             | [failed=10] + [not (failed=10)]*0       | [failed=0 & sent=0]*0.014              |                                            |
| brp               | 3.54                 | 36             | [failed=10] + [not (failed=10)]*0       | [failed=0 & sent=0]*0.05               |                                            |
| brp_finite_family | 3.78                 | 53             | [failed=5] + [not (failed=5)]*0         | [failed<=0 & sent<=0]*0.005            |                                            |
| my_chain_5        | 0.32                 | 5              | [c=1] + [not (c=1)]*0                   | [c=0<br> & x=0]*0.677                  | $p=0.67232$                                |
| grid_small        | 17.2                 | 37             | [a=0 & b=10] + [not (a=0 & b=10)]*0     | [a<=0 & b<=0]*0.006                    | $p=\frac{1}{2^{10}}= 0.0009765625$         |
| gambler           | 0.16                 | 11             | [x=4] + [not (x=4)]*0                   | [x=2 & y=4 & z=0]*0.504                | $p=\frac12=0.5$                            |
| geo0              | 0.09                 | 5              | z                                       | [z=1 & flip=0]*5                       | $\mathbb{E}[z]=5$                          |
| geo0              | 0.11                 | 5              | z                                       | [flip=0]*(z+4)                         | $\mathbb{E}[z]=z+4$                        |
| geo0_obs          | 0.32                 | 5              | z                                       | [flip=0 & z=0]*7                       | Contains observe. $\mathbb{E}[z]=7$        |
| die_conditioning  | 0.41                 | 4              | d1                                      | [flag=0]*5.005                         | Contains observe. $\mathbb{E}[d_1]=5$      |
| PrinSys           | 0.18                 | 1              | [x=2] + [not (x=2)]*0                   | [x=2]*1.005                            | $p=1$                                      |
| RevBin            | TO                   | -              | z                                       | [x=5 & z=3]*11.773                     |                                            |
| sprdwalk          | 0.04                 | 3              | PAST (Positive Almost Sure Termination) | Initial states: [x < n]                |                                            |
| normal            | 76.66                | 2              | [pos=0] + [not (pos=0)]*0               | [flag=0 & coeff=0 & y=0 & pos=0]*0.537 | $p=0.5342$                                 |
| Monty Hall        | 9.05                 | 1              | [win=1] + [not (win=1)]*0               | [flag=0]*0.671                         | Contains an observe statement; $p=\frac23$ |
| Burglar Alarm     | 14.29                | 10             | [burglary=1] + [not (burglary=1)]*0     | [flag=0]*0.034                         | Contains an observe statement              |

## cegispro2 sub-invariants with cdb

In this table we verify the lower bounds given by TSI with cegispro2. TO=5min

| Program name      | cegispro2 total time | Number of CTIs | POSTEXP                             | PROP                                   | Additional details                         |
| ----------------- | -------------------- | -------------- | ----------------------------------- | -------------------------------------- | ------------------------------------------ |
| brp_10            | 171.95               | 43             | [failed=10] + [not (failed=10)]*0   | [failed=0 & sent=0]*0.004              |                                            |
| brp               | 0.37                 | 1              | [failed=10] + [not (failed=10)]*0   | [failed=0 & sent=0]*0                  |                                            |
| brp_finite_family | 0.41                 | 1              | [failed=5] + [not (failed=5)]*0     | [failed<=0 & sent<=0]*                 |                                            |
| my_chain_5        | 3.64                 | 5              | [c=1] + [not (c=1)]*0               | [c=0<br> & x=0]*0.667                  | $p=0.67232$                                |
| grid_small        | 0.31                 | 1              | [a=0 & b=10] + [not (a=0 & b=10)]*0 | [a<=0 & b<=0]*0                        | $p=\frac{1}{2^{10}}= 0.0009765625$         |
| gambler           | TO                   | -              | [x=4] + [not (x=4)]*0               | [x=2 & y=4 & z=0]*0.494                | $p=\frac12=0.5$                            |
| geo0              | TO                   | -              | z                                   | [z=1 & flip=0]*4.954                   | $\mathbb{E}[z]=5$                          |
| geo0              | 0.18                 | 2              | z                                   | [flip=0]*(z+3.952)                     | $\mathbb{E}[z]=z+4$                        |
| geo0_obs          | TO                   | -              | z                                   | [flip=0 & z=0]*6.906                   | Contains observe. $\mathbb{E}[z]=7$        |
| die_conditioning  | TO                   | -              | d1                                  | [flag=0]*4.995                         | Contains observe. $\mathbb{E}[d_1]=5$      |
| PrinSys           | 0.26                 | 1              | [x=2] + [not (x=2)]*0               | [x=2]*0.995                            | $p=1$                                      |
| RevBin            | 0.19                 | 5              | z                                   | [x=5 & z=3]*11.763                     |                                            |
| normal            | TO                   | -              | [pos=0] + [not (pos=0)]*0           | [flag=0 & coeff=0 & y=0 & pos=0]*0.527 | $p=0.5342$                                 |
| Monty Hall        | 239.52               | 1              | [win=1] + [not (win=1)]*0           | [flag=0]*0.661                         | Contains an observe statement; $p=\frac23$ |
| Burglar Alarm     | TO                   | -              | [burglary=1] + [not (burglary=1)]*0 | [flag=0]*0.024                         | Contains an observe statement              |

## Summary

| Program name      | TSI time    | TSI exp                              | t (maximum iterations for TSI) | N (tensor dimensions) | cegispro2 total time | Number of CTIs | cegispro2 total time (cdb sub-invariant) | Number of CTIs (cdb sub-invariant) | Bound aimed to prove with cegispro2 | Proved?          | Additional details                                                                                                                    |
| ----------------- | ----------- | ------------------------------------ | ------------------------------ | --------------------- | -------------------- | -------------- | ---------------------------------------- | ---------------------------------- | ----------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| brp_10            | 3.18        | [0.004, 0.014]                       | 100                            | $10^6$                | 167.71               | 95             | 171.95                                   | 43                                 | [0.004,0.014]                       | yes              |                                                                                                                                       |
| brp               | 0.36 / 3.27 | [-0.005, 0.005]                      | 10 / 100                       | $10^6$                | 3.54                 | 36             | 0.37                                     | 1                                  | [0,0.005]                           | yes              |                                                                                                                                       |
| brp_finite_family | 0.42        | [-0.005, 0.005]                      | 10                             | $10^6$                | 3.78                 | 53             | 0.41                                     | 1                                  | [0,0.005]                           | yes              |                                                                                                                                       |
| my_chain_5        | 0.26        | [0.667, 0.677]                       | 10                             | $10^6$                | 0.32                 | 5              | 3.64                                     | 5                                  | [0.667,0.677]                       | yes              | $p=0.67232$                                                                                                                           |
| grid_small        | 0.63        | [-0.003, 0.006]                      | 20                             | $10^6$                | 17.2                 | 37             | 0.31                                     | 1                                  | [0,0.006]                           | yes              | $p=\frac{1}{2^{10}}= 0.0009765625$                                                                                                    |
| gambler           | 0.88        | \[0.494, 0.504\]                     | 20                             | $10^6$                | 0.16                 | 11             | TO                                       | -                                  | [0.494,0.504]                       | upper bound only | $p=\frac12=0.5$                                                                                                                       |
| geo0              | 0.64        | \[4.954, 4.964\]                     | 10                             | $10^6$                | 0.09                 | 5              | TO                                       | -                                  | [4.954,5]                           | upper bound only | $\mathbb{E}[z]=5$                                                                                                                     |
| geo0              | 0.64        | \[3.952, 3.962\]                     | 20                             | $10^6$                | 0.11                 | 5              | 0.18                                     | 2                                  | [z+3.952,z+4]                       | yes              | $\mathbb{E}[z]=z+4$; TSI exp represents z_final - z_initial. Initially z is a tensor of random values in $[0,10^6]$                   |
| geo0_obs          | 0.65        | [6.906, 6.916]                       | 20                             | $10^6$                | 0.32                 | 5              | TO                                       | -                                  | [6.906,7]                           | upper bound only | Contains observe. $\mathbb{E}[z]=7$                                                                                                   |
| die_conditioning  | 0.04        | [4.997, 5.007]                       | -                              | $10^6$                | 0.41                 | 4              | TO                                       | -                                  | [4.995,5.005]                       | upper bound only | Contains observe. $\mathbb{E}[d_1]=5$                                                                                                 |
| PrinSys           | 0.41        | [0.995, 1.005]                       | 10                             | $10^6$                | 0.18                 | 1              | 0.26                                     | 1                                  | [0.995,1.005]                       | yes              | $p=1$                                                                                                                                 |
| RevBin            | 0.33        | [11.763, 11.773]                     | 10                             | $10^6$                | TO                   | -              | 0.19                                     | 5                                  | [11.763,11.773]                     | lower bound only |                                                                                                                                       |
| sprdwalk          | 0.31        | [2.365,2.375]; lower prob = 0.986897 | 10                             | $10^6$                | 0.04                 | 3              | -                                        | -                                  | -                                   | -                | **TSI**: exp represents expected runtime. Lower prob is term/(term+live). **cegispro2**: used to prove PAST (initial states: [x < n]) |
| normal            | 0.02        | [0.527, 0.537]                       | -                              | $10^6$                | 76.66                | 2              | TO                                       | -                                  | [0.527,0.537]                       | upper bound only | $p=0.5342$                                                                                                                            |
| Monty Hall        | 0.39        | [0.661, 0.671]                       | -                              | $10^6$                | 9.05                 | 1              | 239.52                                   | 1                                  | [0.661,0.671]                       | yes              | Contains an observe statement; $p=\frac23$                                                                                            |
| Burglar Alarm     | 0.08        | [0.024, 0.034]                       | -                              | $10^6$                | 14.29                | 10             | TO                                       | -                                  | [0.024,0.034]                       | upper bound only | Contains an observe statement                                                                                                         |

## Explanation of exact probabilities / expected values

### my_chain_5

0 to 4 increments of x, each with probability $1 - \frac15$ and one c:=1 assignment, with probability $\frac15$
$$\frac15\sum_{i=0}^{4}\left(1 - \frac15\right)^i$$ 
### grid_small

10 consecutive increments of b, each with probability $\frac{1}{2^{10}}$ and no increment of a.

### gambler

For each winning execution, there is a symmetric losing execution. So the probability of winning must be $\frac12$

### geo0 / geo0_obs

Expected value of geometric distribution

### die_conditioning

36 possible combinations of the two dice. Sum of ten can be obtained only by 4+6, 5+5, 6+4. So: $$\mathbb{E}[d_1 | d_1 + d_2 = 10] = 4 \cdot \frac13 + 5 \cdot \frac13 + 6 \cdot \frac13 = 5$$
### PrinSys

Clearly, when starting in a state that violates the loop guard, the final state is the same as the initial. So $p=1$

### normal

The probability can be calculated by means of convolutions. In particular this Matlab code provides the exact probability $p=0.5342$:

```Matlab
probs = repelem(1/11,11);

conv1 = conv2(probs,probs);

c2 = conv2(conv1,probs);

disp(sum(c2(1:16)));
```

### Monty Hall

It is well known that $p=\frac23$
