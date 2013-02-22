import os
import sys
import numpy as np
import h5py

if __name__ == '__main__':
    # Parse arguments
    args = sys.argv[1:]
    dataset_name = args.pop(0)
    num_dims = int(args.pop(0))
    full_sizes = np.array([int(args.pop(0)) for i in range(num_dims)])[::-1]  # Matlab HDF5!
    input_files = args[:-1]
    output_path = args[-1]

    print "Reassembling", " ".join(input_files)
    print "Into", output_path

    # Create temporary output
    temp_path = output_path + '_partial'
    outf = h5py.File(temp_path, 'w')
    out_dataset = None

    # Loop over inputs
    for path in input_files:
        print "Reading", path, "..."
        infile = h5py.File(path)
        original_coords = infile['original_coords'][...]
        diced_data = infile[dataset_name]

        if out_dataset is None:
            # Chunk by block size
            full_sizes = [(s if s > 0 else diced_data.shape[idx]) for idx, s in enumerate(full_sizes)]
            chunksize = np.array(diced_data.shape)
            out_dataset = \
                outf.create_dataset(dataset_name,
                                    full_sizes,
                                    dtype=diced_data.dtype,
                                    chunks=tuple(chunksize),
                                    shuffle=True,
                                    compression='gzip')

        # compute destination coordinates
        lovals = original_coords[:num_dims]
        hivals = original_coords[num_dims:]
        dst_coords = [np.s_[lo:hi] for lo, hi in zip(lovals, hivals)][::-1]  # Matlab HDF5 reorders coords
        out_dataset[tuple(dst_coords)] = diced_data
    outf.close()

    # move to final location
    if os.path.exists(output_path):
        os.unlink(output_path)

    os.rename(temp_path, output_path)
    print "Success"
