#pragma once

#include <cmath>

#define MAX_DIST ((double) 50.)
#define MIN_T ((double) 0.001)
#define DT ((double) 0.001)
#define MAX_STEPS ((double) MAX_DIST / DT)

#define SCALE_FACTOR ((double) 10.00)
#define FRAC ((double) (1) / (MAX_STEPS))
#define CONSTANT_FACTOR ((double) pow( SCALE_FACTOR , FRAC ))
