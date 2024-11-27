from sklearn.preprocessing import MaxAbsScaler


def scaler(x):
    abs_scaler = MaxAbsScaler()
    abs_scaler.fit(x)
    scaled_x = abs_scaler.transform(x)
    return scaled_x