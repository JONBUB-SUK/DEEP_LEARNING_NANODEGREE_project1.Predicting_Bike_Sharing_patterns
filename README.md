# UDACITY_DEEP_LEARNING_NANODEGREE_project1.Predicting_Bike_Sharing_patterns

# Introduction

## 1. Purpose




## 2. Rubric points

1. Car should perfectly drive at least 1 cycle


## 3. Input data given by simulator / Output data for simulator

1. Input

For every "telemetry"
- CTE
- speed
- steering angle

2. Ouput

For every "telemetry", I can send
- steering angle
- throttle





# Background Learning


### 1. PID control

To control steering angle, we can calculate its angle by

alpha = -Kp * CTE -Ki * delta(CTE) - Kd * sum(CTE)

It is reason why being called PID control

There are 3 coefficient Kp, Ki, Kd

1) P means "Propotional"

It affect to steering to turn as much as you are apart from target position

2) I means "Integral"

It is important when my car allignment is not good so it cannot go straight even though steering is straight

3) D means "Differential"

It can prevent oscillation of car

When using only P term, the car will oscillate inevitably


<img src="./images/pid_control.jpg" width="400">



# Content Of This Repo
- ```src``` a directory with the project code
	- ```main.cpp``` : communicate with simulator, reads in data, calls a function in PID.h to drive
	- ```PID.h``` : header file of PID
	- ```PID.cpp``` : have functions have to be used in main.cpp



# Flow

## 1. Preparing Data

Before making code, I made flow chart to check total flow and check what function will be need

<img src="./images/flow_chart_main.png" width="800">


```python
#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
```


# Conclusion & Discussion

### 1. About parameter tuning



