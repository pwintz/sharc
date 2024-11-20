

# Simulation Batching
This Section describes the data representation for simulations executed by SHARC when using batching. 

## Simulation Batch With No Misses

<!-- We use two arrays* of data to store the information. It is useful to have complete data that has a row for time  -->
|`row`|`time`|`i_time_step`|`k_sample`| `x` | `u`                    | `pending_computation` |
|-----|------|-------------|----------|-----|------------------------|-----------------------|
| $0$ |$t_0$ | $0$         | $0$      |$x_0$| $u_0$                  |   $\text{pc}_0$       |
| $1$ |$t_1$ | $0$         | $1$      |$x_1$| $u_0$                  |   $\text{pc}_0$       |
| $2$ |$t_1$ | $1$         | $1$      |$x_1$| $u_1 := \text{pc}_0.u$ |   $\text{pc}_1$       |
| $3$ |$t_2$ | $1$         | $2$      |$x_2$| $u_1$                  |   $\text{pc}_1$       |
| $4$ |$t_2$ | $2$         | $2$      |$x_2$| $u_2 := \text{pc}_1.u$ |   $\text{pc}_2$       |
| $5$ |$t_3$ | $2$         | $3$      |$x_3$| $u_2$                  |   $\text{pc}_2$       |
| $6$ |$t_3$ | $3$         | $3$      |$x_3$| $u_3 := \text{pc}_2.u$ |   $\text{pc}_3$       |
| $7$ |$t_4$ | $3$         | $4$      |$x_4$| $u_3$                  |   $\text{pc}_3$       |
| $8$ |$t_4$ | $4$         | $4$      |$x_4$| $u_4 := \text{pc}_3.u$ |   $\text{pc}_4$       |
| $9$ |$t_5$ | $4$         | $5$      |$x_5$| $u_4$                  |   $\text{pc}_4$       |

In this case, since there are no misses, the next batch start with the following data:
|`row`|`time`|`i_time_step`|`k_sample`| `x` | `u`                    | `pending_computation` |
|-----|------|-------------|----------|-----|------------------------|-----------------------|
|$10$ |$t_5$ | $5$         | $5$      |$x_5$| $u_5 := \text{pc}_4.u$ |   $\text{pc}_5$       |

And would continue from there.

The initial values for the new batch in this case are 

- State: $x_5$
- $i_time_step = 5$
- $u_5 = pc_4.u$

## Simulation With A Missed Compution

Consider, instead, a case where a computation is missed, say at `i_step_index=2`.
Then, all of the data for  `i_time_step ≤ 2` is OK (up to and including `row = 5`). 
The data afterwards is invalid, however.

|`row`|`time`|`i_time_step`|`k_sample`| `x` | `u`                                                         |`pending_computation`  |
|-----|------|-------------|----------|-----|-------------------------------------------------------------|-----------------------|
| $0$ |$t_0$ | $0$         | $0$      |$x_0$| $u_0$                                                       |  $\text{pc}_0$        |
| $1$ |$t_1$ | $0$         | $1$      |$x_1$| $u_0$                                                       |  $\text{pc}_0$        |
| $2$ |$t_1$ | $1$         | $1$      |$x_1$| $u_1 := \text{pc}_0.u$                                      |  $\text{pc}_1$        |
| $3$ |$t_2$ | $1$         | $2$      |$x_2$| $u_1$                                                       |  $\text{pc}_1$        |
| $4$ |$t_2$ | $2$         | $2$      |$x_2$| $u_2 := \text{pc}_1.u$                                      |  $\text{pc}_2$ (LONG) |
| $5$ |$t_3$ | $2$         | $3$      |$x_3$| $u_2$                                                       |  $\text{pc}_2$        |
| $6$ |$t_3$ | $3$         | $3$      |$x_3$| $u_3 := u_2$  ($\text{pc}_2$ not finished yet!)             |  $\text{pc}_3$        |
| $7$ |$t_4$ | $3$         | $4$      |$x_4$| $u_3$                                                       |  $\text{pc}_3$        |

Thus, if we are running batches in parallel, we should restart from row 6 (`i_time_step = 3`, `sample_time_index=3`), using $x = x_3$ (already computed) and `pending_computation`=$\text{pc}_2$ until $\text{pc}_2$ completes.

|`row`|`time`|`i_time_step`|`k_sample`| `x` | `u`                                 |`pending_computation`                 |
|-----|------|-------------|----------|-----|-------------------------------------|--------------------------------------|
| $6$ |$t_3$ | $3$         | $3$      |$x_3$| $u_3 := u_2$                        |  $\text{pc}_3 := \text{pc}_2$        |
| $7$ |$t_4$ | $3$         | $4$      |$x_4$| $u_3$                               |  $\text{pc}_3$                       |
| $8$ |$t_4$ | $4$         | $4$      |$x_4$| $u_3 := \text{pc}_2.u$              |  $\text{pc}_4$                       |
| $9$ |$t_5$ | $4$         | $5$      |$x_5$| $u_3$                               |  $\text{pc}_4$                       |

The initial values for the new batch in this case are 

- State: $x_3$
- `i_time_step` = 3
- $u_3 = u_2$
- `pending_computation` $= pc_2$

<!-- 

|`row`|`time`|`i_time_step`|`k_sample`| `x` | `u`                    | `pending_computation` |
|-----|------|-------------|----------|-----|------------------------|-----------------------|
| $0$ |$t_0$ | $0$         | $0$      |$x_0$| $u_0$                  |   $\text{pc}_0$       |
| $1$ |$t_1$ | $0$         | $1$      |$x_1$| $u_0$                  |   $\text{pc}_0$       |
| $2$ |$t_1$ | $1$         | $1$      |$x_1$| $u_1 := \text{pc}_0.u$ |   $\text{pc}_1$       |
| $3$ |$t_2$ | $1$         | $2$      |$x_2$| $u_1$                  |   $\text{pc}_1$       |
| $4$ |$t_2$ | $2$         | $2$      |$x_2$| $u_2 := \text{pc}_1.u$ |   $\text{pc}_2$ (MISS)|
| $5$ |$t_3$ | $2$         | $3$      |$x_3$| $u_2$                  |   $\text{pc}_2$       |
| $6$ |$t_3$ | $3$         | $3$      |$x_3$| $u_3 := u_2$           |   $\text{pc}_2$       |
| $7$ |$t_4$ | $3$         | $4$      |$x_4$| $u_3$                  |   $\text{pc}_2$       | -->




<!-- 
| `time_step` | `t_start` | `t_end` | `pending_computation` |
|-------------|-----------|---------|-----------------------|
| 0           |   $t_0$   |  $t_1$  |   $\text{pc}_0$       |
| 1           |   $t_1$   |  $t_2$  |   $\text{pc}_1$       |
| 2           |   $t_2$   |  $t_3$  |   $\text{pc}_2$       |
| 3           |   $t_3$   |  $t_4$  |   $\text{pc}_3$       |
| 4           |   $t_4$   |  $t_5$  |   $\text{pc}_4$       |


The
|`row_index`|`time`|`time_step`|`sample_time_index`| `x` | `u`   |
|-----------|------|-----------|-------------------|-----|-------|
| $4$       |$t_2$ | $2$       | $2$               |$x_2$| $u_2$ |
| $5$       |$t_3$ | $2$       | $3$               |$x_3$| $u_2$ |
| $6$       |$t_3$ | $3$       | $3$               |$x_3$| $u_3$ |
| $7$       |$t_4$ | $3$       | $4$               |$x_4$| $u_3$ | |
| $8$       |$t_4$ | $4$       | $4$               |$x_4$| $u_4$ |
| $9$       |$t_5$ | $4$       | $5$               |$x_5$| $u_4$ |

*We don't actually use a single array to store all of this data, but thinking of it as a single array is useful for keeping track of how the data is indexed. -->