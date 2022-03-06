import numpy as np


def affine1(x):
    w, h = x.shape[1], x.shape[0]
    _x = np.zeros(shape=(2 * h - 1, w))
    for col in range(w):
        _x[col:col + h, col] = x[:, col]
    x = _x
    return x


def affine2(x):
    w, h = x.shape[1], x.shape[0]
    _x = np.zeros(shape=(2 * h - 1, w))
    for col in range(w):
        _x[-h - col:2 * h - 1 - col, col] = x[:, col]
    x = _x
    return x


def result(state):
    for i in range(state.shape[0] - 4):
        if np.sum(np.sum(state[i:i + 5], axis=0) == 5) > 0:
            return True, 1., 100., { "status": "WIN" }
        elif np.sum(np.sum(state[i:i + 5], axis=0) == -5) > 0:
            return True, -1., -1., { "status": "LOSE" }
    for j in range(state.shape[1] - 4):
        if np.sum(np.sum(state[:, j:j + 5], axis=1) == 5) > 0:
            return True, 1., 100., { "status": "WIN" }
        elif np.sum(np.sum(state[:, j:j + 5], axis=1) == -5) > 0:
            return True, -1., -1., { "status": "LOSE" }
    state_affine1 = affine1(state)
    for i in range(state_affine1.shape[0] - 4):
        if np.sum(np.sum(state_affine1[i:i + 5], axis=0) == 5) > 0:
            return True, 1., 100., { "status": "WIN" }
        elif np.sum(np.sum(state_affine1[i:i + 5], axis=0) == -5) > 0:
            return True, -1., -1., { "status": "LOSE" }
    for j in range(state_affine1.shape[1] - 4):
        if np.sum(np.sum(state_affine1[:, j:j + 5], axis=1) == 5) > 0:
            return True, 1., 100., { "status": "WIN" }
        elif np.sum(np.sum(state_affine1[:, j:j + 5], axis=1) == -5) > 0:
            return True, -1., -1., { "status": "LOSE" }
    state_affine2 = affine2(state)
    for i in range(state_affine2.shape[0] - 4):
        if np.sum(np.sum(state_affine2[i:i + 5], axis=0) == 5) > 0:
            return True, 1., 100., { "status": "WIN" }
        elif np.sum(np.sum(state_affine2[i:i + 5], axis=0) == -5) > 0:
            return True, -1., -1., { "status": "LOSE" }
    for j in range(state_affine2.shape[1] - 4):
        if np.sum(np.sum(state_affine2[:, j:j + 5], axis=1) == 5) > 0:
            return True, 1., 100., { "status": "WIN" }
        elif np.sum(np.sum(state_affine2[:, j:j + 5], axis=1) == -5) > 0:
            return True, -1., -1., { "status": "LOSE" }
    return False, 0, -.1, { "status": "PLAY" }


if __name__ == '__main__':
    state = np.asarray([
        [0, -1, -1, -1, -1, -1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, -1],
        [0, 0, 1, 0, 0, -1, 0],
        [0, 0, 1, 1, -1, 0, 0],
        [0, 0, 1, -1, 1, 0, 0],
        [0, 0, -1, 0, 0, 1, 0]
    ])

    print(result(state))
