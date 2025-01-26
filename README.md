This repository contains the experiments for my thesis: A comparative analysis of verification techniques for probabilistic programs.
For more details regarding the tools that I have used, you can visit:
- https://github.com/moves-rwth/cegispro2
- https://github.com/Luisa-unifi/probabilistic_programming/blob/main/TSI.py

## Summary

The value of eps is 0.005. This allows to obtain a delta which is less than 0.001 (often much less then 0.001), with the only exception of *die_conditioning* which produces a delta of approximately 0.015. This exception is justified by the fact that there is an observation in the program witnessing a rare event ($p\simeq 0.083$). Nonetheless, the TSI exp captures the true expected value.
The programs that do not contain a while loop, do not have a maximum number of iterations, this is indicated with a dash in the corresponding cell.

| Program name         | $t_{TSI}$     | $t_{cegispro2}$ | $I$                | t   | \|S'\| | True expected value  |
| -------------------- | ------------- | --------------- | ------------------ | --- | ------ | -------------------- |
| brp\_10              | **3.90**      | 183.66          | [0.004, 0.019]     | 100 | 74     | -                    |
| brp                  | 3.27          | **0.44**   | [0, 1]             | 100 | 6      | -                    |
| my\_chain\_5         | **0.54** | 3.96            | [0.667, 0.677]     | 20  | 10     | $p=0.67232$          |
| grid\_small          | **0.63** | 17.51           | [0, 0.006]         | 20  | 38     | $p=\frac{1}{2^{10}}$ |
| gambler              | **0.86** | 0.43 *          | [0.493, 0.509]     | 20  | 3      | $p=\frac12=0.5$      |
| geo0                 | **2.94** | 0.15 *          | [4.997, 5.007] *   | 80  | 5      | $\mathbb{E}[z]=5$    |
| **geo0_obs**         | **2.88** | 0.14 *          | [6.998, 7.008] *   | 80  | 5      | $\mathbb{E}[z]=7$    |
| **die_conditioning** | **0.04** | 0.41 *          | [4.997, 5.007]     | -   | 4      | $\mathbb{E}[d_1]=5$  |
| PrinSys              | **0.41** | 0.44            | [0.995, 1.005]     | 10  | 2      | $p=1$                |
| RevBin               | 0.33          | **0.32**   | [12.996, 13.006] * | 100 | 6      | $\mathbb{E}[z]=13$   |
| normal               | **0.02** | 316.18          | [0.527, 0.537]     | -   | 2      | $p=0.5342$           |
| **Monty Hall**       | **0.39** | 9.05 *          | [0.661, 0.671]     | -   | 2      | $p=\frac23$          |
| **Burglar Alarm**    | **0.08** | 14.29 *         | [0.024, 0.034]     | -   | 10     | $p=0.029$            |

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

### Burglar alarm

Precise value taken from the original TSI paper.

## RevBin


Since initially ```x=5``` and ```z=3``` the support for ```z``` is $\mathbb{N} \setminus \{0,1,\dots, 7\}$. This is because to make ```x=0```, one has to increment ```z``` at least five times. It is now easy to see that $$Pr(Z = z) = \left(\frac12\right)^{z-3} \binom{z-4}{z-8}$$ since ```z``` represents the execution's length + 3. Obviously an execution of length ```z - 3```  has probability $$\left(\frac12\right)^{z-3}$$. The number of these execution is the binomial coefficient reported in the formula. An explanation of this is reported in the next paragraph.
If we map the execution of a decrement to the character 'D' and the execution of the skip statement to the character 'S' we can easily denote all computations of fixed length ```z - 3```. The last instruction must be a decrement, since that is what makes the guard false, so the words must end with the character 'D'. The remaining ```z - 4``` characters must contain 4 decrements. The remaining ```z - 8``` characters must be all 'S'. The order of these is arbitrary so we can represent the number of such words as the above binomial coefficient. Notice that such term is equivalent to $$\binom{z-4}{z-8}$$. Also notice that the formula is well defined as the domain of ```z``` consists of the naturals greater or equal to 8.
The sum seems to converge to 13, this has been verified numerically.
