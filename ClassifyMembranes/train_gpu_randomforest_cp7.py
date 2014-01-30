############################################################
# GPU Implementation of Random Forest Classifier - Training
# v0.1
# Seymour Knowles-Barley
############################################################
# Based on c code from:
# http://code.google.com/p/randomforest-matlab/
# License: GPLv2
############################################################

import numpy as np
import sys
import h5py
import glob
import mahotas
import subprocess
import os

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray

gpu_randomforest_train_source = """
#include "curand_kernel.h"

#define NODE_TERMINAL -1
#define NODE_TOSPLIT  -2
#define NODE_INTERIOR -3

__device__ void movedata() {
}

__device__ void sampledata(const int nclass, const int* nsamples, const int* samplefrom,
	const int maxnsamples, int* bagstart, curandState_t *randstate)
{
	//Select random samples
	int iclass, isamp;
	for (iclass=0; iclass < nclass; ++iclass) {
		for (isamp=0; isamp < nsamples[iclass]; ++isamp) {
			bagstart[isamp + iclass*maxnsamples] = curand(randstate) % samplefrom[iclass];
		}
	}
}

__device__ void sortbagbyx(
	const float *baggedxstart, int totsamples, int mdim, int featurei, int *bagstart, int ndstart, int ndend, int *tempbagstart)
{
	//Sort elements of bagstart (from ndstart to ndend) according to x values
	//Write results into bagstart
	int length = ndend-ndstart+1;
	if (length == 1)
	{
		return;
	}
	int xstart = featurei * totsamples;
	int *inbag = bagstart;
	int *outbag = tempbagstart;

	//For-loop merge sort
	int i = 1;
	int start1, start2, end1, end2, p1, p2, output;
	while (i < length)
	{

		for (start1 = ndstart; start1 <= ndend; start1 += i*2)
		{
			end1 = start1 + i - 1;
			start2 = start1 + i;
			end2 = start2 + i - 1;
			p1 = start1; p2 = start2;
			output = start1;
			while (p1 <= end1 && p1 <= ndend && p2 <= end2 && p2 <= ndend && output <= ndend)
			{
				if (baggedxstart[xstart + inbag[p1]] < baggedxstart[xstart + inbag[p2]])
				{
					outbag[output] = inbag[p1];
					++p1;
				}
				else
				{
					outbag[output] = inbag[p2];
					++p2;
				}
				++output;
			}
			while (p1 <= end1 && p1 <= ndend)
			{
				outbag[output] = inbag[p1];
				++p1;
				++output;
			}
			while (p2 <= end2 && p2 <= ndend)
			{
				outbag[output] = inbag[p2];
				++p2;
				++output;
			}
		}

		//swap for next run
		if (inbag == bagstart)
		{
			inbag = tempbagstart;
			outbag = bagstart;
		}
		else
		{
			inbag = bagstart;
			outbag = tempbagstart;
		}

		//Loop again with larger chunks
		i *= 2;

	}

	//Copy output to bagstart (if necessary)
	if (inbag == tempbagstart)
	{
		for (p1 = ndstart; p1 <= ndend; ++p1)
		{
			bagstart[p1] = tempbagstart[p1];
		}
	}

}

__device__ void findBestSplit(
	const float *baggedxstart, const int *baggedclassstart, int mdim, int nclass, int *bagstart,
	int totsamples, int k, int ndstart, int ndend, int *ndendl,
	int *msplit, float *gini_score, float *best_split, int *best_split_index, bool *isTerminal,
	int mtry, int idx, int maxTreeSize, int *classpop, float* classweights,
	curandState_t *randstate,
	int *wlstart, int *wrstart, int *dimtempstart, int *tempbagstart)
{
	//Compute initial values of numerator and denominator of Gini
	float gini_n = 0.0;
	float gini_d = 0.0;
	float gini_rightn, gini_rightd, gini_leftn, gini_leftd;
	int ctreestart = k * nclass + nclass * idx * maxTreeSize;
	int i;
	for (i = 0; i < nclass; ++i)
	{
		gini_n += classpop[i + ctreestart] * classpop[i + ctreestart];
		gini_d += classpop[i + ctreestart];
	}
	float gini_crit0 = gini_n / gini_d;

	//start main loop through variables to find best split
	float gini_critmax = -1.0e25;
	float crit;
	int trynum, featurei;
	int maxfeature = mdim;

	for (i = 0; i < mdim; ++i)
	{
		dimtempstart[i] = i;
	}

	*msplit = -1;

	//for (trynum = 0; trynum < 1; ++trynum)
	for (trynum = 0; trynum < mtry && trynum < mdim; ++trynum)
	{
		//Choose a random feature
		i = curand(randstate) % maxfeature;
		featurei = dimtempstart[i];
		dimtempstart[i] = dimtempstart[maxfeature-1];
		dimtempstart[maxfeature-1] = featurei;
		--maxfeature;

		//Sort according to this feature
		sortbagbyx(baggedxstart, totsamples, mdim, featurei, bagstart, ndstart, ndend, tempbagstart);

		//Split on numerical predictor featurei
		gini_rightn = gini_n;
		gini_rightd = gini_d;
		gini_leftn = 0;
		gini_leftd = 0;
		for (i = 0; i < nclass; ++i)
		{
			wrstart[i] = classpop[i + ctreestart];
			wlstart[i] = 0;
		}
		int splitpoint;
		int splitxi;
		float split_weight, thisx, nextx;
		int split_class;
		int ntie = 1;
		//Loop through all possible split points
		for (splitpoint = ndstart; splitpoint <= ndend-1; ++splitpoint)
		{
			//Get split details
			splitxi = bagstart[splitpoint];
			//Determine class based on index and nsamples vector
			split_class = baggedclassstart[splitxi]-1;
			split_weight = classweights[split_class];
			//Update neumerator and demominator
			gini_leftn += split_weight * (2 * wlstart[split_class] + split_weight);
			gini_rightn += split_weight * (-2 * wrstart[split_class] + split_weight);
			gini_leftd += split_weight;
			gini_rightd -= split_weight;
			wlstart[split_class] += split_weight;
			wrstart[split_class] -= split_weight;

			//Check if the next value is the same (no point splitting)
			thisx = baggedxstart[splitxi + totsamples * featurei];
			nextx = baggedxstart[bagstart[splitpoint+1] + totsamples * featurei];
			if (thisx != nextx)
			{
				//Check if either node is empty (or very small to allow for float errors)
				if (gini_rightd > 1.0e-5 && gini_leftd > 1.0e-5)
				{
					//Check the split
					crit = (gini_leftn / gini_leftd) + (gini_rightn / gini_rightd);
					if (crit > gini_critmax)
					{
						*best_split = (thisx + nextx) / 2;
						*best_split_index = splitpoint;
						gini_critmax = crit;
						*msplit = featurei;
						*ndendl = splitpoint;
						ntie = 1;
					}
					else if (crit == gini_critmax)
					{
						++ntie;
						//Break ties at random
						if ((curand(randstate) % ntie) == 0)
						{
							*best_split = (thisx + nextx) / 2;
							*best_split_index = splitpoint;
							gini_critmax = crit;
							*msplit = featurei;
							*ndendl = splitpoint;
						}
					}
				}
			}
		} // end splitpoint for
	} // end trynum for

	if (gini_critmax < -1.0e10 || *msplit == -1)
	{
		//We could not find a suitable split - mark as a terminal node
		*isTerminal = true;
	}
	else if (*msplit != featurei)
	{
		//Resort for msplit (if necessary)
		sortbagbyx(baggedxstart, totsamples, mdim, *msplit, bagstart, ndstart, ndend, tempbagstart);
	}
	*gini_score = gini_critmax - gini_crit0;

}

extern "C" __global__ void trainKernel(
	const float *x, int n, int mdim, int nclass,
	const int *classes, const int *classindex,
	const int *nsamples, const int *samplefrom,
	int maxnsamples, 
	unsigned long long seed, unsigned long long sequencestart,
	int ntree, int maxTreeSize, int mtry, int nodeStopSize,
	int *treemap, int *nodestatus, float *xbestsplit, 
	int *bestvar, int *nodeclass, int *ndbigtree,
	int *nodestart, int *nodepop,
	int *classpop, float *classweights,
	int *weight_left, int *weight_right,
	int *dimtemp, int *bagspace, int *tempbag, float *baggedx, int *baggedclass)
{
// Optional arguments for debug (place after xbestsplit): int *nbestsplit, float *bestgini,
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	//Make sure we don't overrun
	if (idx < ntree) {
		//Init random number generators (one for each thread)
		curandState_t state;
		curand_init(seed, sequencestart + idx, 0, &state);

		int i,j,k,cioffset,bioffset;

		int totsamples = 0;
		for (i = 0; i < nclass; ++i){
			totsamples += nsamples[i];
		}

		//Choose random samples for all classes
		int *bagstart = bagspace + idx * nclass * maxnsamples;
		int *tempbagstart = tempbag + idx * nclass * maxnsamples;
		float *baggedxstart = baggedx + idx * mdim * totsamples;
		int *baggedclassstart = baggedclass + idx * totsamples;
		//TODO: offset weightleft, weightright and dimtemp !
		sampledata(nclass, nsamples, samplefrom, maxnsamples, bagstart, &state);

		//Remove gaps and index into x (instead of into class)
		k = 0;
		cioffset = 0;
		bioffset = 0;
		for (i = 0; i < nclass; ++i){
			for (j = 0; j < nsamples[i]; ++j) {
				//Move memory into local block?
				int xindex = classindex[bagstart[j + i * maxnsamples] + cioffset];
				int dimindex;
				for (dimindex = 0; dimindex < mdim; ++dimindex){
					baggedxstart[j + bioffset + totsamples * dimindex] = x[xindex + n * dimindex];
				}
				baggedclassstart[j + bioffset] = classes[xindex];
				bagstart[k] = j + bioffset;
				++k;
			}
			cioffset += samplefrom[i];
			bioffset += nsamples[i];
			classpop[i + idx * nclass * maxTreeSize] = nsamples[i];
		}

		//Wipe other values
		for (;k < nclass * maxnsamples; ++k) {
			bagstart[k] = -1;
		}

		int ndstart, ndend, ndendl;
		int msplit, best_split_index;
		float best_split, gini_score;

		//Repeat findbestsplit until the tree is complete
		int ncur = 0;
		int treeoffset1 = idx * maxTreeSize;
		int treeOffset2 = idx * 2 * maxTreeSize;
		nodestart[treeoffset1] = 0;
		nodepop[treeoffset1] = totsamples;
		nodestatus[treeoffset1] = NODE_TOSPLIT;

		for (k = 0; k < maxTreeSize-2; ++k) {
			//Check for end of tree
			if (k > ncur || ncur >= maxTreeSize - 2) break;
			//Skip nodes we don't need to split
			if (nodestatus[treeoffset1+k] != NODE_TOSPLIT) continue;

			/* initialize for next call to findbestsplit */
			ndstart = nodestart[treeoffset1 + k];
			ndend = ndstart + nodepop[treeoffset1 + k] - 1;
			bool isTerminal = false;
			gini_score = 0.0;
			best_split_index = -1;

			findBestSplit(baggedxstart, baggedclassstart, mdim, nclass, bagstart, totsamples, k, ndstart, ndend, &ndendl,
				&msplit, &gini_score, &best_split, &best_split_index, &isTerminal, mtry, idx, maxTreeSize, classpop, classweights, 
				&state, weight_left + nclass * idx, weight_right + nclass * idx, dimtemp + mdim * idx, tempbagstart);

			if (isTerminal) {
				/* Node is terminal: Mark it as such and move on to the next. */
				nodestatus[k] = NODE_TERMINAL;
				//bestvar[treeoffset1 + k] = 0;
				//xbestsplit[treeoffset1 + k] = 0;
				continue;
			}

			// this is a split node - prepare for next round
			bestvar[treeoffset1 + k] = msplit + 1;
			//bestgini[treeoffset1 + k] = gini_score;
			xbestsplit[treeoffset1 + k] = best_split;
			//nbestsplit[treeoffset1 + k] = best_split_index;
			nodestatus[treeoffset1 + k] = NODE_INTERIOR;
			//varUsed[msplit - 1] = 1;
			//tgini[msplit - 1] += decsplit;

			int leftk = ncur + 1;
			int rightk = ncur + 2;
			nodepop[treeoffset1 + leftk] = ndendl - ndstart + 1;
			nodepop[treeoffset1 + rightk] = ndend - ndendl;
			nodestart[treeoffset1 + leftk] = ndstart;
			nodestart[treeoffset1 + rightk] = ndendl + 1;

			// Check for terminal node conditions
			nodestatus[treeoffset1 + leftk] = NODE_TOSPLIT;
			if (nodepop[treeoffset1 + leftk] <= nodeStopSize) {
				nodestatus[treeoffset1 + leftk] = NODE_TERMINAL;
			}

			nodestatus[treeoffset1 + rightk] = NODE_TOSPLIT;
			if (nodepop[treeoffset1 + rightk] <= nodeStopSize) {
				nodestatus[treeoffset1 + rightk] = NODE_TERMINAL;
			}

			//Calculate class populations
			int nodeclass = 0;
			int ctreestart_left = leftk * nclass + idx * nclass * maxTreeSize;
			int ctreestart_right = rightk * nclass + idx * nclass * maxTreeSize;
			for (i = ndstart; i <= ndendl; ++i) {
				nodeclass = baggedclassstart[bagstart[i]]-1;
				classpop[nodeclass + ctreestart_left] += classweights[nodeclass];
			}
			for (i = ndendl+1; i <= ndend; ++i) {
				nodeclass = baggedclassstart[bagstart[i]]-1;
				classpop[nodeclass + ctreestart_right] += classweights[nodeclass];
			}

			for(i = 0; i < nclass; ++i)
			{
				if (classpop[i + ctreestart_left] == nodepop[treeoffset1 + leftk])
				{
					nodestatus[treeoffset1 + leftk] = NODE_TERMINAL;
				}
				if (classpop[i + ctreestart_right] == nodepop[treeoffset1 + rightk])
				{
					nodestatus[treeoffset1 + rightk] = NODE_TERMINAL;
				}
			}

			//Update treemap offset (indexed from 1 rather than 0)
			treemap[treeOffset2 + k*2] = ncur + 2;
			treemap[treeOffset2 + 1 + k*2] = ncur + 3;
			ncur += 2;

		}

		//Tidy up
		//TODO: Check results - should not be necessary to go up to maxTreeSize
		ndbigtree[idx] = ncur+1;
		//ndbigtree[idx] = maxTreeSize;
		for(k = maxTreeSize-1; k >= 0; --k)
		{
			//if (nodestatus[treeoffset1 + k] == 0)
			//	--ndbigtree[idx];
			if (nodestatus[treeoffset1 + k] == NODE_TOSPLIT)
				nodestatus[treeoffset1 + k] = NODE_TERMINAL;
		}

		//Calculate prediction for terminal nodes
		for (k = 0; k < maxTreeSize; ++k)
		{
			treeoffset1 = idx * maxTreeSize;
			if (nodestatus[treeoffset1 + k] == NODE_TERMINAL)
			{
				int toppop = 0;
				int ntie = 1;
				for (i = 0; i < nclass; ++i)
				{
					int ctreeoffset = k * nclass + idx * nclass * maxTreeSize;
					if (classpop[i + ctreeoffset] > toppop)
					{
						nodeclass[treeoffset1 + k] = i+1;
						toppop = classpop[i + ctreeoffset];
					}
					//Break ties at random
					if (classpop[i + ctreeoffset] == toppop)
					{
						++ntie;
						if ((curand(&state) % ntie) == 0)
						{
							nodeclass[treeoffset1 + k] = i+1;
							toppop = classpop[i + ctreeoffset];
						}
					}
				}
			}
		}

		//ndbigtree[idx] = idx;

	}

}
"""

input_image_folder1 = 'D:\\dev\\datasets\\Cerebellum\\Classifiers\\P7_Data'
raw_image_suffix = '.png'
input_image_suffix = '_train.png'
input_features_suffix = '_rhoana_features.h5'
output_path = 'D:\\dev\\datasets\\Cerebellum\\Classifiers\\P7_Data\\rhoana_forest_cp7_2class.h5'

features_prog = 'D:\\dev\\Rhoana\\rhoana\\ClassifyMembranes\\x64\\Release\\compute_features.exe'

# Prep the gpu function
gpu_train = nvcc.SourceModule(gpu_randomforest_train_source, no_extern_c=True).get_function('trainKernel')

# Load training data
files = sorted( glob.glob( input_image_folder1 + '\\*' + input_image_suffix ) )
#files = files + sorted( glob.glob( input_image_folder2 + '\\*' + input_image_suffix ) )

#2 Class
class_colors = [[255,0,0], [0,255,0]]
#class_colors = [[255,85,255], [255,255,0]]

# 3 Class
#class_colors = [[255,0,0], [0,255,0], [0,0,255]]
#class_colors = [[255,85,255], [255,255,0], [0,255,255]]

# class_colors = [0, 1]

nclass = len(class_colors)

training_x = np.zeros((0,0), dtype=np.float32)
training_y = np.zeros((0,1), dtype=np.int32)

print 'Found {0} training images.'.format(len(files))

# Loop through all images
for file in files:
	training_image = mahotas.imread(file)

	# Load the features
	image_file = file.replace(input_image_suffix, raw_image_suffix)
	features_file = file.replace(input_image_suffix, input_features_suffix)

	if not os.path.exists(features_file):
		print "Computing features:", features_prog, image_file, features_file
		subprocess.check_call([features_prog, image_file, features_file], env=os.environ)

	f = h5py.File(features_file, 'r')

	nfeatures = len(f.keys())

	for classi in range(nclass):

		this_color = class_colors[classi]

		# Find pixels for this class
		# class_indices = np.nonzero(np.logical_and(
		# 	training_image[:,:,this_color] > training_image[:,:,(this_color + 1) % 3],
		# 	training_image[:,:,this_color] > training_image[:,:,(this_color + 2) % 3]))

		class_indices = np.nonzero(np.logical_and(
			training_image[:,:,0] == this_color[0],
			training_image[:,:,1] == this_color[1],
			training_image[:,:,2] == this_color[2]))

		# Add features to x and classes to y

		training_y = np.concatenate((training_y, np.ones((len(class_indices[0]), 1), dtype=np.int32) * (classi + 1)))

		train_features = np.zeros((nfeatures, len(class_indices[0])), dtype=np.float32)

		for i,k in enumerate(f.keys()):
			feature = f[k][...]
			train_features[i,:] = feature[class_indices[0], class_indices[1]]

		if training_x.size > 0:
			training_x = np.concatenate((training_x, train_features), axis=1)
		else:
			training_x = train_features

	f.close()

for classi in range(nclass):
	print 'Class {0}: {1} training pixels.'.format(classi, np.sum(training_y == classi + 1))

# Train on GPU
ntree = np.int32(512)
mtry = np.int32(np.floor(np.sqrt(training_x.shape[0])))
#nsamples = np.ones((1,nclass), dtype=np.int32) * (training_x.shape[1] / nclass)
nsamples = np.ones((1,nclass), dtype=np.int32) * 1000
classweights = np.ones((1,nclass), dtype=np.float32)

# Sanity check
assert(training_x.shape[1] == training_y.shape[0])

# Random number seeds
seed = np.int64(42)
sequencestart = np.int64(43)

samplefrom = np.zeros((nclass), dtype=np.int32)
maxTreeSize = np.int32(2 * np.sum(nsamples) + 1)
nodeStopSize = np.int32(1)

for classi in range(nclass):
	samplefrom[classi] = np.sum(training_y == (classi + 1))

maxnsamples = np.max(nsamples)
classindex = -1 * np.ones((np.max(samplefrom) * nclass), dtype=np.int32)

cioffset = 0
for classi in range(nclass):
	classindex[cioffset:cioffset + samplefrom[classi]] = np.nonzero(training_y == (classi + 1))[0]
	cioffset = cioffset + samplefrom[classi]

bagmem = -1 * np.ones((ntree, maxnsamples * nclass), dtype=np.int32)
d_bagspace = gpuarray.to_gpu(bagmem)
d_tempbag = gpuarray.to_gpu(bagmem)
bagmem = None

d_treemap = gpuarray.zeros((long(ntree * 2), long(maxTreeSize)), np.int32)
d_nodestatus = gpuarray.zeros((long(ntree), long(maxTreeSize)), np.int32)
d_xbestsplit = gpuarray.zeros((long(ntree), long(maxTreeSize)), np.float32)
#d_nbestsplit = gpuarray.zeros((long(ntree), long(maxTreeSize)), np.int32)
#d_bestgini = gpuarray.zeros((long(ntree), long(maxTreeSize)), np.float32)
d_bestvar = gpuarray.zeros((long(ntree), long(maxTreeSize)), np.int32)
d_nodeclass = gpuarray.zeros((long(ntree), long(maxTreeSize)), np.int32)
d_ndbigtree = gpuarray.zeros((long(ntree), 1), np.int32)
d_nodestart = gpuarray.zeros((long(ntree), long(maxTreeSize)), np.int32)
d_nodepop = gpuarray.zeros((long(ntree), long(maxTreeSize)), np.int32)
d_classpop = gpuarray.zeros((long(ntree), long(maxTreeSize*nclass)), np.int32)
d_classweights = gpuarray.to_gpu(classweights)
d_weight_left = gpuarray.zeros((long(ntree), long(nclass)), np.int32)
d_weight_right = gpuarray.zeros((long(ntree), long(nclass)), np.int32)
d_dimtemp = gpuarray.zeros((long(ntree), long(training_x.shape[0])), np.int32)

d_baggedx = gpuarray.zeros((long(np.sum(nsamples)*training_x.shape[0]), long(ntree)), np.float32)
d_baggedclass = gpuarray.zeros((long(ntree), long(np.sum(nsamples))), np.int32)

d_training_x = gpuarray.to_gpu(training_x)
d_training_y = gpuarray.to_gpu(training_y)
d_classindex = gpuarray.to_gpu(classindex)
d_nsamples = gpuarray.to_gpu(nsamples)
d_samplefrom = gpuarray.to_gpu(samplefrom)

threadsPerBlock = 32
block = (32, 1, 1)
grid = (int(ntree / block[0] + 1), 1)

gpu_train(d_training_x, np.int32(training_x.shape[1]), np.int32(training_x.shape[0]), np.int32(nclass),
	d_training_y, d_classindex, d_nsamples, d_samplefrom,
	np.int32(maxnsamples), seed, sequencestart, np.int32(ntree), np.int32(maxTreeSize), np.int32(mtry), np.int32(nodeStopSize),
	d_treemap, d_nodestatus, d_xbestsplit,
	d_bestvar, d_nodeclass, d_ndbigtree,
	d_nodestart, d_nodepop,
	d_classpop, d_classweights,
	d_weight_left, d_weight_right,
	d_dimtemp, d_bagspace, d_tempbag, d_baggedx, d_baggedclass,
	block=block, grid=grid)

treemap = d_treemap.get()
nodestatus = d_nodestatus.get()
xbestsplit = d_xbestsplit.get()
bestvar = d_bestvar.get()
nodeclass = d_nodeclass.get()
ndbigtree = d_ndbigtree.get()

# Save results
out_hdf5 = h5py.File(output_path, 'w')
out_hdf5['/forest/treemap'] = treemap
out_hdf5['/forest/nodestatus'] = nodestatus
out_hdf5['/forest/xbestsplit'] = xbestsplit
out_hdf5['/forest/bestvar'] = bestvar
out_hdf5['/forest/nodeclass'] = nodeclass
out_hdf5['/forest/ndbigtree'] = ndbigtree

out_hdf5['/forest/nrnodes'] = maxTreeSize
out_hdf5['/forest/ntree'] = ntree
out_hdf5['/forest/nclass'] = nclass
out_hdf5['/forest/classweights'] = classweights
out_hdf5['/forest/mtry'] = mtry

out_hdf5.close()
