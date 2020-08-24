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
    
    
    def precompute_distances(self, r=40):
        print("start precomputing values in distance array")
        for i in range(self.size[0]):
            radius = r
            for j in range(self.size[1]):
                if self._map[i, j] <= 0.0:
                    obstacle = 0
                    n = 0
                    for x in range(max(0, i - radius), min(i+radius, self.size[0])):
                        for y in range(max(0, j - radius), min(j+radius, self.size[1])):
                            if np.linalg.norm([x-i, y-j]) < radius / 2:
                                n += 1
                                if self._map[x, y] < 0:
                                    obstacle += 1
                    self._map[i, j] = -25 * (obstacle / n)
                else:
                    self._map[i, j] = radius
                    for x in range(max(0, i - radius), min(i+radius, self.size[0])):
                        for y in range(max(0, j - radius), min(j+radius, self.size[1])):
                            if self._map[x, y] < 0.0:
                                d = np.sqrt((x-i)**2 + (y-j)**2)
                                if d < self._map[i, j]:
                                    self._map[i, j] = d
        print("finished.")
                    
        
    def get_value(self, x, y):
        try:
            return self._map[int(x+self.offset[0]), int(y+self.offset[1])]
        except IndexError:
            pass
        return -5000
    
    
    def heatmap(self, plot_range=None):
        if plot_range is None:
            plot_range = np.arange(0 + offset[0], self._map.shape[0] + offset[0])
        data = []
        for i in plot_range:
            for j in plot_range:
                item = {"x" : i, "y": j, "value": self.get_value(i, j)}
                data.append(item)
        obstacle_df = pd.DataFrame(data)
        heat = sns.heatmap(obstacle_df.pivot(index="y", columns="x", values="value"), cmap=self.cmap, center=0, cbar=False)
        heat.invert_yaxis()
        
        heat.set_xticks([])
        heat.set_yticks([])
        heat.set_xlabel(None)
        heat.set_ylabel(None)
        return heat
    
    def save(self, filename):
        np.save(filename, self._map)
        
def save_labyrinth(bar_length, difficulty="hard"):
    print(f"creating: obstacles/labyrinth_{difficulty}_{bar_length}")
    values = np.ones((200, 200))
    values[0,:] = -5
    values[-1,:] = -5
    values[:,0] = -5
    values[:,-1] = -5
    values[60:70,:bar_length] = -5
    if difficulty == "hard":
        values[95:105,-bar_length:] = -5
        values[130:140,:bar_length] = -5
    elif difficulty == "easy":
        values[130:140,-bar_length:] = -5
    else:
        print("no preset for given difficulty")
        return

    obstacles = ObstacleMap(value=values)
    obstacles.precompute_distances()
    obstacles.save(f"obstacles/labyrinth_{difficulty}_{bar_length}")
    print("done.")
    
def save_double_gap(bar_length, difficulty="easy", gaps=2):
    print(f"creating: obstacles/gaps_{gaps}_{difficulty}_{bar_length}")
    values = np.ones((200, 200))
    values[0,:] = -5
    values[-1,:] = -5
    values[:,0] = -5
    values[:,-1] = -5
    
    if difficulty == "hard" or difficulty == "medium":
        values[60:70,:bar_length] = -5
    
    if difficulty == "hard":
        values[130:140,-bar_length:] = -5

        
    length = int(200 / (2 * gaps -1))
    y = length
    while y + length < 200:
        values[95:105,y:y+length] = -5
        y += 2*length

    obstacles = ObstacleMap(value=values)
    obstacles.precompute_distances()
    obstacles.save(f"obstacles/gaps_{gaps}_{difficulty}_{bar_length}")
    print("done.")
    

if __name__ == "__main__":
    for gaps in range(2, 6):
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty == "easy":
                save_double_gap(60, gaps=gaps, difficulty=difficulty)
                continue
            for bar_length in [60, 80, 90, 95,100,105, 110, 120, 140]:
                save_double_gap(bar_length, gaps=gaps, difficulty=difficulty)
