#ifndef _MODEL_BINDINGS_
#define _MODEL_BINDINGS_

#include <vector>

using Point = std::vector<int>;

std::vector<Point> floodFill(Point& lowerBound, 
                          Point& upperBound);

#endif
