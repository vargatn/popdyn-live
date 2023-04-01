import numpy as np

def dpad(arr):
    padded = np.pad(arr, pad_width=1, mode="edge")[:, :, 1:-1]
def pad(arr):
    padded = np.pad(arr, pad_width=1, mode="edge")[:, :, 1:-1]
    return padded

def dcenter_cut(arr):
    return arr[1:-1, 1:-1, :]

def center_cut(arr):
    return arr[1:-1, 1:-1]

def null_edges(arr, axis, depth=1, value=0):
    arr[:depth, :, axis] = value
    arr[:, :depth, axis] = value
    arr[:, -depth:, axis] = value
    arr[-depth:, :, axis] = value
    return arr

def calc_shifted(inmap):
    """shift up, down, left, right"""

    omap = []
    tmp = np.roll(inmap, -1, axis=0)
    tmp[-1:, :] = 0
    omap.append(tmp)

    tmp = np.roll(inmap, 1, axis=0)
    tmp[:1, :] = 0
    omap.append(tmp)

    tmp = np.roll(inmap, -1, axis=1)
    tmp[:, -1:] = 0
    omap.append(tmp)

    tmp = np.roll(inmap, 1, axis=1)
    tmp[:, :1] = 0
    omap.append(tmp)
    return np.array(omap)

def calc_unshifted(inmap):
    omap = []

    tmp = np.roll(inmap[0, :, :], 1, axis=0)
    tmp[:1, :] = 0
    omap.append(tmp)

    tmp = np.roll(inmap[1, :, :], -1, axis=0)
    tmp[-1:, :] = 0
    omap.append(tmp)

    tmp = np.roll(inmap[2, :, :], 1, axis=1)
    tmp[:, :1] = 0
    omap.append(tmp)

    tmp = np.roll(inmap[3, :, :], -1, axis=1)
    tmp[:, -1:] = 0
    omap.append(tmp)

    return np.array(omap)


class CellSpace(object):
    def __init__(self, initial_maps, layers, rules):
        """
        Cell automata
        """
        self.canvas = copy.deepcopy(initial_maps)
        self.layers = layers
        self.rules = rules

        self.padded_canvas = flow.pad(self.canvas)
        self.shape_canvas = self.canvas.shape
        self.shape_canvas_padded = self.padded_canvas.shape

        self._make_indexes()

        self.shifted_canvases = self.shift()

        self.reset_maps()

    def _make_indexes(self):
        ishape = self.shape_canvas[:2]  # only the map part of the stack
        self.imap = np.arange(ishape[0] * ishape[1]).reshape(ishape)
        ishape = self.shape_canvas_padded[:2]  # only the map part of the stack
        self.imap_padded = np.arange(ishape[0] * ishape[1]).reshape(ishape)
        self.imap_center = flow.center_cut(self.imap_padded)
        self.imap_shifted = [flow.center_cut(tmp) for tmp in flow.calc_shifted(self.imap_padded)]
        self.imap_unshifted = np.array([
            self.imap_shifted[1],
            self.imap_shifted[0],
            self.imap_shifted[3],
            self.imap_shifted[2],
        ])

    def reset_maps(self):
        self.maps = []

    def update_maps(self):
        self.maps.append(self.canvas)

    def shift(self):
        shifted_canvas = np.zeros(shape=((4,) + self.canvas.shape))
        for i, ii in enumerate(self.imap_shifted):
            for j in np.arange(self.canvas.shape[-1]):
                shifted_canvas[i, :, :, j] = self.padded_canvas[:, :, j].flat[ii]

        return shifted_canvas

    def step(self):
        """
        Apply the rules to the canvas (in order), the rules must return the residual change!!!!

        """

        new_canvas = np.zeros(self.canvas.shape)
        for rule in self.rules:
            new_canvas += rule.evolve(self)

        self.canvas = new_canvas
        self.padded_canvas = flow.pad(self.canvas)
        self.shifted_canvas = self.shift()

    def flow(self, nstep=100, savestep=5):
        """
        a series of steps
        """
        self.reset_maps()
        self.update_maps()
        for i in np.arange(nstep):
            print(str(i) + ' out of ' + str(nstep - 1), end="\n")
            self.step()
            if i % savestep == 0:
                self.update_maps()