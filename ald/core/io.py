import h5py
import os


def get_h5attr(file, attr):
    with h5py.File(file, "r") as f:
        return f.attrs[attr]


def get_h5dataset(file, path):
    with h5py.File(file, "r") as f:
        return f[path][...]


class Result:
    def __init__(self, h5file):
        if not h5py.is_hdf5(h5file):
            raise TypeError("expects a hdf5 file.")
        # keep reference to file
        self.h5file = h5file
        # extract root attributes from hdf5 dataset
        with h5py.File(self.h5file, "r") as f:
            for attr, value in f.attrs.items():
                setattr(self, attr, value)

    def get_data(self, path):
        return get_h5dataset(self.h5file, path)

    def get_attr(self, attr):
        return get_h5attr(self.h5file, attr)

    def get_x(self, frame=0):
        """Get x positions"""
        path = "config/{}/".format(frame)
        x = self.get_data(os.path.join(path, "x"))
        # get time too
        t = self.get_data(os.path.join(path, "t"))
        return x, t

    def get_y(self, frame=0):
        """Get y positions"""
        path = "config/{}/".format(frame)
        y = self.get_data(os.path.join(path, "y"))
        # get time too
        t = self.get_data(os.path.join(path, "t"))
        return y, t

    def get_theta(self, frame=0):
        """Get y positions"""
        path = "config/{}/".format(frame)
        theta = self.get_data(os.path.join(path, "theta"))
        # get time too
        t = self.get_data(os.path.join(path, "t"))
        return theta, t
