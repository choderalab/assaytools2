import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn

def plot_rv(rv, n_sample = 10000, style = 'pca'):
    """plt some distribution.

    Parameters
    ----------
    rv :
        
    n_sample :
         (Default value = 10000)
    style :
         (Default value = 'pca')

    Returns
    -------

    """
    samples = rv.sample(n_sample).numpy()
    print(rv.name)
    if 'LogNormal' in str(rv.name):
        samples = np.log(samples)
        print(samples)

    if samples.ndim == 1 or samples.shape[1] == 1:
        samples = samples.flatten()
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111)
        hist, _ = np.histogram(samples, bins=50)
        x_axis = np.array(range(50))
        ax.plot(x_axis, np.true_divide(hist, hist.sum()))
        return ax, None

    elif style == 'sep':
        assert len(samples.shape) == 2
        dims = samples.shape[1]
        figs = []
        for dim in range(dims):
            fig = plt.figure()
            figs.append(fig)
            hist, _ = np.histogram(samples[:, dim], bins=50)
            x_axis = np.array(range(50))
            ax = fig.add_subplot(111)
            ax.plot(x_axis, np.true_divide(hist, hist.sum()))
        return figs, None

    elif style == 'pca':
        from matplotlib import cm
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111, projection='3d')
        ax._axis3don = False
        samples = rv.sample(n_sample).numpy()
        pca = sklearn.decomposition.PCA(2)
        samples_transformed = pca.fit_transform(samples)
        hist, x_edges, y_edges = np.histogram2d(samples_transformed[:, 0], samples_transformed[:, 1], bins=50)
        hist = np.true_divide(hist, hist.sum())
        x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1])
        ax.plot_wireframe(x_pos, y_pos, hist, cmap=cm.coolwarm)
        return ax, pca



def plot_est(points, rv, rv_ax, style = 'pca', pca=None):
    """plot the estimation points

    Parameters
    ----------
    points :
        
    rv :
        
    rv_ax :
        
    style :
         (Default value = 'pca')
    pca :
         (Default value = None)

    Returns
    -------

    """
    zs = np.array([])
    for idx in range(points.shape[0]):
        if 'LogNormal' in rv.name:
            zs = np.append(zs, np.exp(np.sum(rv.log_prob(tf.constant(np.exp(points[idx]), dtype=tf.float32)))))
        else:
            zs = np.append(zs, np.exp(np.sum(rv.log_prob(tf.constant(points[idx], dtype=tf.float32)))))

    if points.ndim == 1 or samples.shape[1] == 1:
        xs = points.flatten()
        rv_ax.plot(xs, zs, 'x')
        return rv_ax

    if style == 'sep':
        raise NotImplementedError

    if style == 'pca':
        size = points.shape[1]
        points = pca.transform(points)
        xs = points[:, 0]
        ys = points[:, 1]
        rv_ax.plot(xs, ys, zs, 'r-')
        return rv_ax

def plot_all(points, rv, n_sample = 10000):
    """plot the distribution and the sample points

    Parameters
    ----------
    points :
        
    rv :
        
    n_sample :
         (Default value = 10000)

    Returns
    -------

    """

    samples = rv.sample(n_sample).numpy()

    if 'LogNormal' in str(rv.name):
        samples = np.log(samples)

    zs = np.array([])
    for idx in range(points.shape[0]):
        if 'LogNormal' in rv.name:
            zs = np.append(zs, np.exp(np.sum(rv.log_prob(tf.constant(np.exp(points[idx]), dtype=tf.float32)))))
        else:
            zs = np.append(zs, np.exp(np.sum(rv.log_prob(tf.constant(points[idx], dtype=tf.float32)))))


    if samples.ndim == 1 or samples.shape[1] == 1:
        samples = samples.flatten()
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111)
        hist, _ = np.histogram(samples, bins=50)
        x_axis = np.array(range(50))
        ax.plot(x_axis, np.true_divide(hist, hist.sum()))
        xs = points.flatten()
        ax.plot(xs, zs, 'x')


    else:
        from matplotlib import cm
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111, projection='3d')
        ax._axis3don = False
        samples = rv.sample(n_sample).numpy()
        pca = sklearn.decomposition.PCA(2)
        samples_transformed = pca.fit_transform(samples)
        hist, x_edges, y_edges = np.histogram2d(samples_transformed[:, 0], samples_transformed[:, 1], bins=50)
        hist = np.true_divide(hist, hist.sum())
        x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1])
        ax.plot_wireframe(x_pos, y_pos, hist, cmap=cm.coolwarm)
        size = points.shape[1]
        points = pca.transform(points)
        xs = points[:, 0]
        ys = points[:, 1]
        # ax.plot(xs, ys, zs, 'r-')
