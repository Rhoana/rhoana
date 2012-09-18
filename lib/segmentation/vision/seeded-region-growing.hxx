#pragma once
#ifndef ANDRES_VISION_SEEDED_REGION_GROWING
#define ANDRES_VISION_SEEDED_REGION_GROWING

#include <vector>
#include <queue>
#include <algorithm> // max
#include <map>

#include "marray.hxx"

namespace andres {
namespace vision {

template<class T> 
void seededRegionGrowing(const View<unsigned char>&, View<T>&);

template<class T> 
void seededRegionGrowingWithPaths(const View<unsigned char>&, View<T>&, Marray<unsigned char>&);

namespace detail {
   template<class T> 
   inline bool isAtSeedBorder(const View<T>& labeling, const size_t index);
}

template<class T>
void seededRegionGrowing
(
	const View<unsigned char>& vol,
	View<T>& labeling
)
{
	// define 256 queues, one for each gray level.
	std::vector<std::queue<size_t> > queues(256);

	// add each unlabeled pixels which is adjacent to a seed
	// to the queue corresponding to its gray level
	for(size_t j = 0; j < labeling.size(); ++j) {
		if(detail::isAtSeedBorder<T>(labeling, j)) {
			queues[vol(j)].push(j);
		}
	}

	// flood
	unsigned char grayLevel = 0;
	for(;;) {
		while(!queues[grayLevel].empty()) {
			// label pixel and remove from queue
			size_t j = queues[grayLevel].front();
			queues[grayLevel].pop();

			// add unlabeled neighbors to queues
			// left, upper, and front voxel
    		size_t coordinate[3];
			labeling.indexToCoordinates(j, coordinate);
			for(unsigned short d = 0; d<3; ++d) {
				if(coordinate[d] != 0) {
					--coordinate[d];
					if(labeling(coordinate) == 0) {
						size_t k;
                        labeling.coordinatesToIndex(coordinate, k);
						unsigned char queueIndex = std::max(vol(coordinate), grayLevel);
						labeling(k) = labeling(j); // label pixel
						queues[queueIndex].push(k);
					}
					++coordinate[d];
				}
			}
			// right, lower, and back voxel
			for(unsigned short d = 0; d<3; ++d) {
				if(coordinate[d] < labeling.shape(d)-1) {
					++coordinate[d];
					if(labeling(coordinate) == 0) {
						size_t k;
                        labeling.coordinatesToIndex(coordinate, k);
						unsigned char queueIndex = std::max(vol(coordinate), grayLevel);
						labeling(k) = labeling(j); // label pixel
						queues[queueIndex].push(k);
					}
					--coordinate[d];
				}
			}
		}
		if(grayLevel == 255) {
			break;
		}
		else {
			queues[grayLevel] = std::queue<size_t>(); // free memory
			++grayLevel;
		}
	}
}

/* \todo this code needs debugging
template<class T>
void seededRegionGrowingWithPaths
(
	const View<unsigned char>& vol,
	View<T>& labeling, 
	Marray<unsigned char>& directions,
	std::map<std::pair<T, T>, std::pair<size_t, size_t> >& adjacency
)
{
    directions.reshape(vol.shapeBegin(), vol.shapeEnd());    

	// define 256 queues, one for each gray level.
	std::vector<std::queue<size_t> > queues(256);

	// add each unlabeled pixels which is adjacent to a seed
	// to the queue corresponding to its gray level
	for(size_t j = 0; j < labeling.size(); ++j) {
		if(detail::isAtSeedBorder<T>(labeling, j)) {
			queues[vol(j)].push(j);
		}
	}

	// flood
	unsigned char grayLevel = 0;
	for(;;) {
		while(!queues[grayLevel].empty()) {
			// label pixel and remove from queue
			size_t j = queues[grayLevel].front();
			queues[grayLevel].pop();

			// add unlabeled neighbors to queues
			// left, upper, and front voxel
    		size_t coordinate[3];
			labeling.indexToCoordinates(j, coordinate);
			for(unsigned short d = 0; d<3; ++d) {
				if(coordinate[d] != 0) {
					--coordinate[d];
					if(labeling(coordinate) == 0) {
						size_t k;
                        labeling.coordinatesToIndex(coordinate, k);
						unsigned char queueIndex = std::max(vol(coordinate), grayLevel);
						labeling(k) = labeling(j); // label pixel
						directions(k) = d+1; // save direction
						queues[queueIndex].push(k);
					}
					else if(labeling(coordinate) != labeling(j)) {
                        size_t k;
						labeling.coordinatesToIndex(coordinate, k);
						if(labeling(j) < labeling(k)) {
							std::pair<T, T> p(labeling(j), labeling(k));
							if(adjacency.count(p) == 0) {
								adjacency[p] = std::pair<size_t, size_t>(j, k);
							}
						}
						else {
							std::pair<T, T> p(labeling(k), labeling(j));
							if(adjacency.count(p) == 0) {
								adjacency[p] = std::pair<size_t, size_t>(k, j);
							}
						}
					}
					++coordinate[d];
				}
			}
			// right, lower, and back voxel
			for(unsigned short d = 0; d<3; ++d) {
				if(coordinate[d] < labeling.shape(d)-1) {
					++coordinate[d];
					if(labeling(coordinate) == 0) {
                        size_t k;
						labeling.coordinatesToIndex(coordinate, k);
						unsigned char queueIndex = std::max(vol(coordinate), grayLevel);
						labeling(k) = labeling(j); // label pixel
						directions(k) = d+4; // save direction
						queues[queueIndex].push(k);
					}
					else if(labeling(coordinate) != labeling(j)) {
                        size_t k;
						labeling.coordinatesToIndex(coordinate, k);
						if(labeling(j) < labeling(k)) {
							std::pair<T, T> p(labeling(j), labeling(k));
							if(adjacency.count(p) == 0) {
								adjacency[p] = std::pair<size_t, size_t>(j, k);
							}
						}
						else {
							std::pair<T, T> p(labeling(k), labeling(j));
							if(adjacency.count(p) == 0) {
								adjacency[p] = std::pair<size_t, size_t>(k, j);
							}
						}
					}
					--coordinate[d];
				}
			}
		}
		if(grayLevel == 255) {
			break;
		}
		else {
			queues[grayLevel] = std::queue<size_t>(); // free memory
			++grayLevel;
		}
	}
}
*/

namespace detail {

template<class T>
inline bool isAtSeedBorder
(
	const View<T>& labeling,
	const size_t index
)
{
	if(labeling(index) == 0) {	
		return false; // not a seed voxel
	}
	else {
		size_t coordinate[3];
		labeling.indexToCoordinates(index, coordinate);
		// check left, upper, and front voxel for zero label
		for(unsigned short d = 0; d<3; ++d) {
			if(coordinate[d] != 0) {
				--coordinate[d];
				if(labeling(coordinate) == 0) {
					return true;
				}
				++coordinate[d];
			}
		}
		// check right, lower, and back voxel for zero label
		for(unsigned short d = 0; d<3; ++d) {
			if(coordinate[d] < labeling.shape(d)-1) {
				++coordinate[d];
				if(labeling(coordinate) == 0) {
					return true;
				}
				--coordinate[d];
			}
		}
		return false;
	}
}

} // namespace detail
} // namespace vision
} // namespace andres

#endif // #ifndef ANDRES_VISION_SEEDED_REGION_GROWING
