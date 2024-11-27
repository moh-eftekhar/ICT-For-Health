from sklearn.cluster import KMeans


def KMeans_runner(x_train, x_test):
    km = KMeans(n_clusters=19, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x_train)
    y_km = km.predict(x_test)
    return y_km
