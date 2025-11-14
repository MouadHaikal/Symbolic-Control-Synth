#include "floodFill.hpp"
#include <queue>
#include <set>


inline bool comparator(Point& point, Point& upperBound) {
    for(int i = 0; i < point.size(); i++) {
        if(point[i] > upperBound[i]) return false;
    }
    return true;
}

std::vector<Point> floodFill(Point &lowerBound, Point &upperBound) {
    std::set<Point> availableCells;
    std::queue<Point> q;
    q.push(lowerBound);

    while(!q.empty()) {
        Point top = q.front();
        q.pop();
        if(availableCells.count(top)) continue;
        availableCells.insert(top);
        int dimension = lowerBound.size();

        for(int msk = 0; msk < (1 << dimension); msk++) {
            Point neighbor = top;
            for(int i = 0; i < dimension; i++) {
                if((msk >> i) & 1) neighbor[i] += 1;
            }
            if(comparator(neighbor, upperBound)) q.push(neighbor);
        }
    }

    std::vector<Point> cells;
    for(Point x : availableCells) cells.push_back(x);

    return cells;
}
