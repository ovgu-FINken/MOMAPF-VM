import numpy as np
import seaborn as sns
import pandas as pd

class ObstacleMap:
    def __init__(self, size=(200, 200), offset=(0, 0), value=None, filename=None):
        if filename is not None:
            value = np.load(filename)
        self.size = size
        self.offset = offset
        self.cmap = sns.cubehelix_palette(dark=0, light=1, rot=-.1, gamma=.25, reverse=True, as_cmap=True)
        if value is not None:
            self._map = value
            self.size = value.shape
        else:
            self._map = np.zeros( size )
    
    
    def precompute_distances(self, r=20):
        print("start precomputing values in distance array")
        current_i = 0
        for i in range(self.size[0]):
            radius = r
            for j in range(self.size[1]):
                if self._map[i, j] <= 0.0:
                    continue
                self._map[i, j] = radius
                for x in range(max(0, i - radius), min(i+radius, self.size[0])):
                    for y in range(max(0, j - radius), min(j+radius, self.size[1])):
                        if self._map[x, y] <= 0:
                            d = np.sqrt((x-i)**2 + (y-j)**2)
                            if d < self._map[i, j]:
                                self._map[i, j] = d
                radius = min(int(self._map[i, j]) + 3, r)
        print("finished.")
                    
        
    def get_value(self, x, y):
        try:
            return self._map[int(x+self.offset[0]), int(y+self.offset[1])]
        except IndexError:
            pass
        return -5
    
    
    def heatmap(self, plot_range=None):
        if plot_range is None:
            plot_range = np.arange(0 + offset[0], self._map.shape[0] + offset[0])
        data = []
        for i in plot_range:
            for j in plot_range:
                item = {"x" : i, "y": j, "value": self.get_value(i, j)}
                data.append(item)
        obstacle_df = pd.DataFrame(data)
        heat = sns.heatmap(obstacle_df.pivot(index="y", columns="x", values="value"), cmap=self.cmap, center=0)
        heat.invert_yaxis()
        return heat
    
    def save(self, filename):
        np.save(filename, self._map)
        

if __name__ == "__main__":
    values = np.ones((200, 200))
    values[0,:] = -5
    values[-1,:] = -5
    values[:,0] = -5
    values[:,-1] = -5
    values[70:-70,95:-95] = -5
    values[95:105,-50:-1] = -5
    values[95:105,30:100] = -5

    obstacles = ObstacleMap(value=values)
    obstacles.precompute_distances()

    obstacles.save("cross.obstacles")