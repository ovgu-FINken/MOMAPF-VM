import dubins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


DUBINS = 1
STRAIGHT = 2
RTR = 3
REED_SHEPP = 4

def waypoints_to_path(waypoints, r=1, step=0.1, r_step=0.1, model=DUBINS):
    path = []
    for wp1, wp2 in zip(waypoints[:-1], waypoints[1:]):
        if model == DUBINS:
            dbp = dubins.shortest_path(wp1, wp2, r)
            path = path + dbp.sample_many(step)[0]
        elif model == RTR or model == STRAIGHT:
            # rotate (1)
            dist = np.linalg.norm(np.array(wp1[:-1]) - np.array(wp2[:-1]))
            x = wp1[0]
            y = wp1[1]
            phi = np.angle(wp1[2])
            if dist > step:
                phi_goal = np.arctan2(wp1[0] - wp2[0], wp1[1] - wp2[1])
            else:
                phi_goal = wp2[2]
            if model == RTR:
                angles = np.arange(phi, phi_goal, r_step)
                for a in angles:
                    path.append( (x, y, a) )
            
            path.append( wp1 )
            # translate
            steps = dist / step
            if steps < 1:
                steps = 1
            dx = (wp2[0] - wp1[0]) / steps
            dy = (wp2[1] - wp1[1]) / steps
            for _ in range(int(steps)):
                x += dx
                y += dy
                path.append( (x, y, phi_goal) )
                
            if model == RTR:
                # rotate (2)
                x = wp2[0]
                y = wp2[1]
                phi = phi_goal
                phi_goal = np.angle(wp2[2])
                angles = np.arange(phi, phi_goal, r_step)
                for a in angles:
                    path.append( (x, y, a) )
                if len(path) < 3:
                    print("OH NO")
                    print(f"{wp1}, {wp2}, {path}, {phi}")
        else:
            print("NO VEHICLE MODEL!")
        
    if model == DUBINS:
        path.append(waypoints[-1])
    return path


def agents_objectives(agents, r=1, step=0.1, model=None, obstacles=None):
    paths = [waypoints_to_path(agent, r=r, step=step, model=model) for agent in agents]
    robustness = robustness_from_paths(paths, obstacles=obstacles)
    lp = [len(path)*step for path in paths]
    return robustness, np.max(lp), np.mean(lp)


def plot_waypoints(wp, color=None, alpha=0.05, marker="o"):
    x = [c[0] for c in wp]
    y = [c[1] for c in wp]
    return plt.plot(x, y, marker, color=color, alpha=alpha)


def wps_to_df(wps):
    data = []
    for i, agent in enumerate(wps):
        for j, wp in enumerate(agent):
            data_i = {
                "x": wp[0],
                "y": wp[1],
                "phi": wp[2],
                "agent" : f"A{i}",
                "time": j
            }
            data.append(data_i)
    return pd.DataFrame(data)


def min_dist_in_configuration(configuration):
    lc = len(configuration)
    m = [np.inf]
    for i in range(1,lc):
        if configuration[i] is None:
            continue
        for j in range(i):
            if configuration[j] is None:
                continue
            d = (configuration[i][0] - configuration[j][0])**2
            d += (configuration[i][1] - configuration[j][1])**2
            m.append(d)
    return np.sqrt(np.min(m))


def robustness_from_paths(paths, obstacles=None):
    d = []
    collision = False
    for path in paths:
        for point in path:
            do = obstacles.get_value(point[0], point[1])
            d.append(do)
            if (do < 0):
                collision = True
    if collision:
        dn = [1.0 for x in d if x < 0]
        return -100 * len(dn)/len(d)
    d = [ np.min(d) ] 
    for configuration in itertools.zip_longest(*paths):
        d.append(min_dist_in_configuration(configuration))
    return np.min(d)


if __name__ == "__main__":
    
    print("running dubins waypoint stitcher")
    wp = [(0, 0, np.pi), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    wp2 = [(3, 0, 0), (1, 0, 0)]
    wp3 = [(3, 4, np.pi), (0, 4, np.pi), (0, 0, 0)]
    
