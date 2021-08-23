import numpy as np


def main(mat):
    trans = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])

    print("Starting mat")
    print(mat)

    print("Translation mat")
    print(trans)

    n_mat, n_sec = trans.shape

    out_mat = np.zeros((n_mat, n_sec))
    for row in range(n_mat):
        a = np.expand_dims(mat[row, :], axis=1)
        a = np.broadcast_to(a, trans.shape) * trans
        out_mat[row, :] = a.sum(axis=0)

    out_mat_2 = np.zeros((n_sec, n_sec))
    for col in range(n_sec):
        b = np.expand_dims(out_mat.T[col, :], axis=1)
        b = np.broadcast_to(b, trans.shape) * trans
        out_mat_2[:, col] = b.sum(axis=0)

    print("Translation result")
    print(out_mat_2)


def pure_np_main(mat):
    trans = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])

    print("Starting mat")
    print(mat)

    print("Translation mat")
    print(trans)

    n_mat, n_sec = trans.shape

    # Translate rows
    mult_shape = (n_mat, n_mat, n_sec)
    a = np.broadcast_to(np.expand_dims(mat, axis=2), mult_shape)
    trans_a = np.broadcast_to(np.expand_dims(trans, axis=1), mult_shape)
    temp = a * trans_a

    # mat is transposed, but we need it this way
    out_mat = temp.sum(axis=0)

    # Translate cols
    mult_shape = (n_mat, n_sec, n_sec)
    b = np.broadcast_to(np.expand_dims(out_mat, axis=2), mult_shape)
    trans_b = np.broadcast_to(np.expand_dims(trans, axis=1), mult_shape)
    temp = b * trans_b
    out_mat_2 = temp.sum(axis=0)

    print("Translation result")
    print(out_mat_2)


def disagg():
    mat = np.ones((3, 3)) * 1000

    trans = np.array([
        [.25, .75, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, .5, .5],
    ])

    print("Starting mat")
    print(mat)

    print("Translation mat")
    print(trans)

    n_mat, n_sec = trans.shape

    # Translate rows
    mult_shape = (n_mat, n_mat, n_sec)
    a = np.broadcast_to(np.expand_dims(mat, axis=2), mult_shape)
    trans_a = np.broadcast_to(np.expand_dims(trans, axis=1), mult_shape)
    temp = a * trans_a

    # mat is transposed, but we need it this way
    out_mat = temp.sum(axis=0)

    # Translate cols
    mult_shape = (n_mat, n_sec, n_sec)
    b = np.broadcast_to(np.expand_dims(out_mat, axis=2), mult_shape)
    trans_b = np.broadcast_to(np.expand_dims(trans, axis=1), mult_shape)
    temp = b * trans_b
    out_mat_2 = temp.sum(axis=0)

    print("Translation result")
    print(out_mat_2)
    print(out_mat_2.sum())


def vector():
    trans = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])

    vec = np.array([8, 2, 8, 8, 5])

    print("Starting mat")
    print(vec)

    print("Translation mat")
    print(trans)

    a = np.broadcast_to(np.expand_dims(vec, axis=1), trans.shape)
    temp = a * trans
    out = temp.sum(axis=0)

    print("Translation result")
    print(out)
    print(out.sum())


if __name__ == '__main__':
    mat = np.random.randint(1, 10, 25).reshape((5, 5))

    print("for version")
    main(mat)
    print('\n\n')

    print("pure numpy")
    pure_np_main(mat)
    print('\n\n')

    print("Disaggregation")
    disagg()
    print('\n\n')

    print("Vector")
    vector()
    print('\n\n')
