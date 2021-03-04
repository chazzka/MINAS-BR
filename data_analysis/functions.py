
def split_atributtes_labels(df, number_labels, labels_ini=True):
    if labels_ini:
        train_set_labels = df.iloc[:,:number_labels]
        train_set_input = df.drop(train_set_labels.columns,axis=1)
    else:
        train_set_labels = df.iloc[:,(number_labels-1):]
        train_set_input = df.drop(train_set_labels.columns,axis=1)
    return train_set_input, train_set_labels