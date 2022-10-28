import argparse
import xarray as xr
from cuml import PCA
import cupy as cp
from tqdm import tqdm
import os

class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.parse_args(open(values).read().split(), namespace)

def climpca():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root-dir', type=str, default='../data/',
                                help="Root directory containing the climate datasets")
    parser.add_argument('--mask-dir', type=str, default='masks/', help="Directory containing the mask datasets")
    parser.add_argument('--train-data-name', type=str, default='train.nc',
                                help="Comma separated list of netCDF files (climate dataset) for training")
    parser.add_argument('--test-data-name', type=str, default='train.nc',
                                help="Comma separated list of netCDF files (climate dataset) for infilling")
    parser.add_argument('--mask-name', type=str, default=None,
                                help="Comma separated list of netCDF files (mask dataset). "
                                     "If None, it extracts the masks from the climate dataset")
    parser.add_argument('--data-type', type=str, default='tas',
                                help="Comma separated list of variable types, "
                                     "in the same order as data-names and mask-names")
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size")
    parser.add_argument('--n-iter', type=int, default=50, help="Number of iterations")
    parser.add_argument('--n-components', type=int, default=100, help="Number of components")
    parser.add_argument('--initial-guess', type=float, default=None, help="Initial guess for missing data")
    parser.add_argument('--optimize-to-ncmax', type=int, default=None, help="Search optimal number of components "
                                                                     "by varying until ncmax")
    parser.add_argument('--output-dir', type=str, default='outputs/',
                                help="Directory where the output files will be stored")
    parser.add_argument('--output-name', type=str, default='output',
                                help="Prefix used for the output filename")
    parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile,
                                help="Load all the arguments from a text file")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    data_train = cp.asarray(xr.open_dataset(args.data_root_dir + "/train/" + args.train_data_name)[args.data_type].values)
    data_train = data_train.reshape(data_train.shape[0], -1)

    ds = xr.open_dataset(args.data_root_dir + "/test/" + args.test_data_name)
    data_test = cp.asarray(ds[args.data_type].values)
    shape = data_test.shape
    data_test = data_test.reshape(shape[0],-1)

    if args.batch_size is None:
        nt = shape[0]
    else:
        nt = args.batch_size
    ns = shape[1] * shape[2]

    assert ns == len(data_train[1]), "Inconsistent train/test data: {}, {}".format(ns, len(data_train[1]))

    nc = shape[0] // nt
    assert nc * nt == shape[0], "The total number of timesteps must be a multiple of the number of chunks"

    print("* Shape of train data:", data_train.shape)
    print("* Shape of test data:", data_test.shape)
    print("* Batch size:", nt)
    print("* Number of chunks:", nc)

    idx = [cp.s_[i * nt:( i + 1 ) * nt] for i in range(nc)]

    mask = []
    if args.mask_name is None:
        for it in range(nc):
            mask.append(cp.isnan(data_test[idx[it]]))
    else:
        mask_val = cp.asarray(xr.open_dataset(args.mask_dir + "/" + args.mask_names)[args.data_type].values)
        assert mask_val.shape == shape
        mask_val = mask_val.reshape(shape[0],-1)
        for it in range(nc):
            mask.append(cp.argwhere(mask_val[idx[it]] == 0))

    # print("* Shape of mask:", *[mask[it].shape for it in range(nc)])

    if args.initial_guess is None:
        data_mean = cp.expand_dims(cp.nanmean(data_train, axis=0), axis=0)
        data_mean = cp.repeat(data_mean, nt, axis=0)
    else:
        data_mean = cp.zeros((nt,ns))
        data_mean[:,:] = args.initial_guess

    print("* Shape of data mean:", data_mean.shape)

    data_tmp = data_test.copy()
    for it in range(nc):
        data_tmp[idx[it]][mask[it]] = data_mean[mask[it]]
    ds[args.data_type].values = data_tmp.get().reshape(-1,*shape[1:])
    ds.to_netcdf(args.output_dir + "/" + args.output_name + "_image.nc")

    data_train = cp.vstack((cp.zeros((nt, ns)), data_train))
    print("* Shape of data stack:", data_train.shape)

    if args.optimize_to_ncmax is None:
        nc_max = args.n_components + 1
    else:
        nc_max = args.optimize_to_ncmax
        data_optim = cp.zeros(shape)

    data_tmp = cp.zeros(shape)
    optim_k = None
    min_err = cp.inf
    max_iter = ( nc_max - args.n_components ) * nc * args.n_iter
    pbar = tqdm(total = max_iter)
    for k in range(args.n_components, nc_max):
        for it in range(nc):

            data_train[:nt] = data_test[idx[it]]
            data_train[:nt][mask[it]] = data_mean[mask[it]]

            pca = PCA(n_components = k)
            for i in range(args.n_iter):
                pbar.update(1)
                pca.fit(data_train)

                trans = pca.transform(data_train)
                image_recon = pca.inverse_transform(trans)
                data_train[:nt][mask[it]] = image_recon[:nt][mask[it]]
            data_tmp[idx[it]] = data_train[:nt].reshape(-1,*shape[1:])

        if args.optimize_to_ncmax is not None:
            rmse = cp.sqrt(((data_test - data_tmp)**2).mean())
            if rmse < min_err:
                min_err = rmse.copy()
                optim_k = k
                data_optim = data_tmp.copy()

    if args.optimize_to_ncmax is None:
        ds[args.data_type].values = data_tmp.get()
    else:
        print("Optimal number of components: ", optim_k)
        ds[args.data_type].values = data_optim.get()

    ds.to_netcdf(args.output_dir + "/" + args.output_name + "_infilled.nc")

if __name__ == "__main__":
    climpca()