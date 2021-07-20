from sklearn.decomposition import PCA


def dim_reduct(data, no_dim=8):
    if data.shape[1] > no_dim:
        pca = PCA(n_components=no_dim, svd_solver='auto')
        return pca.fit_transform(data), lambda d: pca.transform(d)
    else:
        return data, lambda d: d

