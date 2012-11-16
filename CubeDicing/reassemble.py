import os
import sys
import numpy as np
import h5py

if __name__ == '__main__':
    # Parse arguments
    args = sys.argv[1:]
    num_dims = int(args.pop(0))
    full_sizes = np.array([int(args.pop(0)) for i in range(num_dims)])
    halo_sizes = np.array([int(args.pop(0)) for i in range(num_dims)])
    input_files = args[:-1]
    output_path = args[-1]

    print "Reassembling", " ".join(input_files)
    print "Into", output_path
    print "With halos", halo_sizes

    # Create temporary output
    temp_path = output_path + '_partial'
    outf = h5py.File(temp_path, 'w')
    out_dataset = None

    # Loop over inputs
    for path in input_files:
        infile = h5py.File(path)
        diced_output_name = infile['diced_output_name'][...]
        original_coords = infile['original_coords'][...]
        diced_data = infile[diced_output_name]

        if out_dataset is None:
            # Chunk by block size
            chunksize = np.array(diced_data.shape)
            for idx in len(halo_sizes):
                chunksize[idx] -= 2 * halo_sizes[idx]
            out_dataset = \
                outf.create_dataset(diced_output_name,
                                    full_sizes,
                                    dtype=diced_data.dtype,
                                    chunks=tuple(chunksize),
                                    compressed=True)

        # compute destination coordinates
        src_coords = [np.s_[halo:-halo] for halo in halo_sizes][::-1]  # Matlab
        dst_coords = [np.s_[(lo + halo):(lo + sz - halo)][::-1]  # Matlab
                      for lo, sz, halo in zip(original_coords,
                                              diced_data.shape,
                                              halo_sizes)]
        out_dataset[dst_coords] = diced_data[src_coords]

    # move to final location
    if os.path.exists(output_path):
        os.unlink(output_path)

    os.rename(temp_path, output_path)
