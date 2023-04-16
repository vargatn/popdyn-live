import numpy as np


class FlowRule(object):
    def __init__(self, flow_factor=0.5):
        """
        The way this automata is set up is with a walled boundary condition, matter can flow to the edges, but not out of it
        Therefore matter is conserved in the canvas. That's good

        An erosion factor of 1 makes a really unstable checkerboard behaviour, use smaller values for a balanced behaviour,
        """
        self.layers = np.sort(("terrain",))
        self.key = "terrain"
        #         self.needed_keys ["terrain",]
        if flow_factor >= 1:
            raise ValueError("flow_factor must be less than 1, current value is {:.3f}".format(flow_factor))
        self.flow_factor = flow_factor

    def evolve(self, cell):
        height_local = cell.canvases[self.key]
        height_shifted = cell.shifted_canvases[self.key]

        diffs = height_local - height_shifted
        #         diffs[diffs < 0] = 0
        diffs[diffs < diffs.max(axis=0)] = 0.
        #         fractions = np.nan_to_num(diffs / diffs.sum(axis=0))

        outflow = diffs * self.flow_factor  # this is the outgoing amount
        #         print(outflow)
        height_local = self.calc_flows(height_local, outflow, cell)
        resdict = cell.canvases
        resdict.update({self.key: height_local})
        return resdict

    def _calc_flows(self, vmap, outflow, cell):
        shifted_landslide = []
        for i, ii in enumerate(cell.imap_unshifted):
            # the padding here is what restricts the landslide to the frozen boundary condition
            tmp = np.pad(outflow[i], pad_width=1, mode="constant", constant_values=0)
            tmp = tmp.flat[ii]
            shifted_landslide.append(tmp)
        shifted_landslide = np.array(shifted_landslide)
        return shifted_landslide.sum(axis=0)

    def calc_flows(self, vmap, outflow, cell):
        vmap -= outflow.sum(axis=0)
        vmap += self._calc_flows(vmap, outflow, cell)
        return vmap


class RainFallClosed(FlowRule):
    def __init__(self, net_water=1e5, fraction=0.2, seed=10, net_evap=0.0, **kwargs):
        """
        This is adds new rain water to the water canvas

        """
        self.layers = np.sort(("water",))
        self.key = "water"
        self.net_water = net_water
        self.fraction = fraction
        self.net_evap = net_evap
        self.rng = np.random.RandomState(seed)

    def evolve(self, cell):
        water_local = cell.canvases[self.key]

        sshape = water_local.shape
        darea = sshape[0] * sshape[1]
        rainmap = np.ones(shape=sshape) * self.net_water / darea
        water_local += rainmap
        print(rainmap.sum())
        resdict = cell.canvases
        resdict.update({self.key: water_local})
        return resdict


class RainBurst(FlowRule):
    def __init__(self, net_water=1e5, burst_step=40, seed=10, **kwargs):
        """
        This is adds new rain water to the water canvas

        """
        self.layers = np.sort(("water",))
        self.key = "water"
        self.net_water = net_water
        self.burst_step = burst_step
        self.step = 0
        self.rng = np.random.RandomState(seed)

    def evolve(self, cell):
        resdict = cell.canvases

        if self.step % self.burst_step == 0:
            water_local = cell.canvases[self.key]
            sshape = water_local.shape
            darea = sshape[0] * sshape[1]
            rainmap = np.ones(shape=sshape) * self.net_water / darea
            water_local += rainmap
            print(rainmap.sum())
            resdict.update({self.key: water_local})

        self.step += 1
        return resdict

    #     def evolve(self,  cell):
#         water_local = cell.canvases[self.key]

#         # so far this is the evaporation and outflow
# #         if self.net_evap > 0:
# #             water_local -= self.net_evap
# #             water_local[water_local < 0] = 0.
# #         water_local = null_edges(water_local) # this is the flow out of the box
# #         if (water_tmp < 0).sum():
# #             raise ValueError("water should never be negative")

# #         net_rainfall = self.net_water - water_local.sum()
# #         print(net_rainfall)
# #         if net_rainfall > 0:
# #             rmax = 100
# #             rainmap = np.random.random(size=water_local.shape) * rmax
# #             rpiv = rmax * (1 - self.fraction)
# #             rainmap[rainmap <= rpiv] = 0.
# #             ii = rainmap > rpiv
# #             rainmap[ii] = net_rainfall / ii.sum()
# #             print(net_rainfall / ii.sum())

# #         print(water_local.sum(), water_tmp.sum())
# #         print("net_rainfall", net_rainfall)
# #         if (net_rainfall  < 0).sum():
# #             raise ValueError("rainfall should never be negative")
#         water_local += rainmap

# #         print("SUM", water_local.sum(), self.net_water)

#         resdict = cell.canvases
#         resdict.update({self.key: water_local})
#         return resdict

