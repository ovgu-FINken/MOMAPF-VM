import dubins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from enum import IntEnum

class Vehicle(IntEnum):
    DUBINS = 1
    STRAIGHT = 2
    RTR = 3
    REED_SHEPP = 4

class Metric(IntEnum):
    MIN = 1
    MEAN = 2
    MIXED = 3

def waypoints_to_path(waypoints, r=1, step=0.1, r_step=0.1, model=Vehicle.DUBINS):
    path = []
    for wp1, wp2 in zip(waypoints[:-1], waypoints[1:]):
        if model == Vehicle.DUBINS:
            dbp = dubins.shortest_path(wp1, wp2, r)
            path = path + dbp.sample_many(step)[0]
        elif model == Vehicle.RTR or model == Vehicle.STRAIGHT:
            # rotate (1)
            dist = np.linalg.norm(np.array(wp1[:-1]) - np.array(wp2[:-1]))
            x = wp1[0]
            y = wp1[1]
            phi = np.angle(wp1[2])
            if dist > step:
                phi_goal = np.arctan2(wp1[0] - wp2[0], wp1[1] - wp2[1])
            else:
                phi_goal = wp2[2]
            if model == Vehicle.RTR:
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
                
            if model == Vehicle.RTR:
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
        
    if model == Vehicle.DUBINS:
        path.append(waypoints[-1])
    return path


def agents_objectives(agents, r=1, step=0.1, model=None, obstacles=None, metric=None):
    paths = [waypoints_to_path(agent, r=r, step=step, model=model) for agent in agents]
    robustness = robustness_from_paths(paths, obstacles=obstacles, metric=metric)
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

def min_dist_per_agent(configuration):
    lc = len(configuration)
    m = np.full((lc,lc), np.inf)
    for i in range(1, lc):
        if configuration[i] is None:
            continue
        for j in range(i):
            if configuration[j] is None:
                continue
            d = (configuration[i][0] - configuration[j][0])**2
            d += (configuration[i][1] - configuration[j][1])**2
            m[i,j] = d
    m0 = np.min(m, axis=0)
    m1 = np.min(m, axis=1)
    return np.sqrt(np.min([m0, m1], axis=0))

def robustness_from_paths(paths, obstacles=None, metric=None):
    d_obstacles = []
    collision = False
    collision_value = 0
    for agent in paths:
        d_agent = []
        for point in agent:
            do = obstacles.get_value(point[0], point[1])
            d_agent.append(do)
        d_a_min = np.min(d_agent)
        if d_a_min < 0 or collision:
            collision = True
            for value in d_agent:
                if value < 0:
                    collision_value += -1 * 100 / len(d_agent) / len(paths)
        else:
            d_obstacles.append(d_a_min)
    if collision:
        return collision_value 
    d_agents = np.array(d_obstacles) * 2
    for configuration in itertools.zip_longest(*paths):
        d_agents = np.min([d_agents, min_dist_per_agent(configuration)], axis=0)
    if metric is None or metric == Metric.MIN:
        return np.min(d_agents)
    if metric == Metric.MIXED:
        return np.min(d_agents) + np.mean(d_agents) / 1000
    if metric == Metric.MEAN:
        return np.mean(d_agents)
    print("unkown metric")
    return None


if __name__ == "__main__":
    
    print("running dubins waypoint stitcher")
    wp = [(0, 0, np.pi), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    wp2 = [(3, 0, 0), (1, 0, 0)]
    wp3 = [(3, 4, np.pi), (0, 4, np.pi), (0, 0, 0)]
    
