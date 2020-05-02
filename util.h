#pragma once

#include <cmath>

#define MAX_DIST ((double) 50.)
#define MIN_T ((double) 0.001)
#define DT ((double) 0.001)
#define MAX_STEPS ((double) MAX_DIST / DT)

#define SCALE_FACTOR ((double) 10.00)
#define FRAC ((double) (1) / (MAX_STEPS))
#define CONSTANT_FACTOR ((double) pow( SCALE_FACTOR , FRAC ))

#define LOOKATX ((double) 0.) 
#define LOOKATY ((double) 0.)
#define LOOKATZ ((double) 0.)

#define CAMERAX ((double) 0.)
#define CAMERAY ((double) 6.)
#define CAMERAZ ((double) 2.)
