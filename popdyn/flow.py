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
    def __init__(self, initial_maps, rules):
        """
        Cell automata, considers the direct neighbor cells only

        Initialize with a dict of initial maps in the style

        initial_maps = {
            "layer_name": initial map,
        }

        The rules are a similar dictionary of class instances which will be passed


        """
        self.canvases = copy.deepcopy(initial_maps)
        self.layers = list(initial_maps.keys())
        self.padded_canvases = pad(self.canvases)

        self.shape_canvas = self.canvases[self.layers[0]].shape
        self.shape_canvas_padded = self.padded_canvases[self.layers[0]].shape

        self.imap = np.arange(len(self.canvases[self.layers[0]].flatten())).reshape(self.shape_canvas)
        self.imap_padded = np.arange(len(self.padded_canvases[self.layers[0]].flatten())).reshape(
            self.shape_canvas_padded)
        self.imap_center = center_cut(self.imap_padded)
        self.imap_shifted = [center_cut(tmp) for tmp in calc_shifted(self.imap_padded)]
        self.imap_unshifted = [
            self.imap_shifted[1],
            self.imap_shifted[0],
            self.imap_shifted[3],
            self.imap_shifted[2],
        ]

        self.shifted_canvases = self.shift()
        self.rules = rules

        self.maps = {}
        self.reset_maps()

    def update_maps(self):
        for key in self.layers:
            self.maps[key].append(self.canvases[key].copy())

    def reset_maps(self):
        self.maps = {}
        for key in self.layers:
            #             dd = {key: [self.canvases[key],]}
            dd = {key: []}
            self.maps.update(dd)

    def shift(self):
        shifted_canvases = {}
        for key in self.canvases.keys():
            arr = np.array([self.padded_canvases[key].flat[ii] for ii in self.imap_shifted])
            shifted_canvases.update({key: arr})
        return shifted_canvases

    def step(self):

        new_canvases = {}
        for rule in self.rules:
            canvas = rule.evolve(self)
            new_canvases.update(canvas)

        self.canvases = new_canvases
        self.padded_canvases = pad(self.canvases)
        self.shifted_canvases = self.shift()

    def flow(self, nstep=100, savestep=5):
        """
        a series of steps
        """
        self.reset_maps()
        self.update_maps()
        for i in np.arange(nstep):
            #             print(str(i) + ' out of ' + str(nstep-1), end="\r")
            print(str(i) + ' out of ' + str(nstep - 1), end="\n")
            self.step()
            if i % savestep == 0:
                self.update_maps()