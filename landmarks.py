def firstRow():
    landmarks = ['label']
    for val in range(1, 33+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val)]
    return landmarks