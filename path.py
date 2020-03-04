import dubins
import reeds_shepp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import bezier

from enum import IntEnum

class Vehicle(IntEnum):
    DUBINS = 1
    STRAIGHT = 2
    RTR = 3
    REEDS_SHEPP = 4
    BEZIER = 5

class Metric(IntEnum):
    MIN = 1
    MEAN = 2
    MIXED = 3
    
    
def short_angle_range(phi1, phi2, r_step=0.2):
    if np.abs(phi1 - (2*np.pi + phi2)) < np.abs(phi1 - phi2):
        phi2 = 2*np.pi + phi2
    if np.abs(phi1 - (-2*np.pi + phi2)) < np.abs(phi1 - phi2):
        phi2 = -2*np.pi + phi2
        
    if phi1 < phi2:
        return np.arange(phi1, phi2, r_step)
    return np.arange(phi1, phi2, -r_step)
    

def waypoints_to_path(waypoints, r=1, step=0.1, r_step=0.2, model=Vehicle.DUBINS, FIX_ANGLES=False):
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
            phi = wp1[2] % (2 * np.pi)
            if dist > step:
                dx = wp2[0] - wp1[0]
                dy = wp2[1] - wp1[1]
                # as per https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html
                # Note the role reversal: the “y-coordinate” is the first function parameter, the “x-coordinate” is the second.
                phi_goal = np.arctan2(dy, dx)
            else:
                phi_goal = wp2[2]
            if model == Vehicle.RTR:
                for a in short_angle_range(phi, phi_goal, r_step=r_step):
                    path.append( (x, y, a) )
            
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
                phi_goal = wp2[2] % (2 * np.pi)
                for a in short_angle_range(phi, phi_goal, r_step=r_step):
                    path.append( (x, y, a) )
                    
#                if len(path) < 3:
#                    print("OH NO")
#                    print(f"{wp1}, {wp2}, {path}, {phi}")
        elif model==Vehicle.REEDS_SHEPP:
            part = []
            sample = reeds_shepp.path_sample(wp1, wp2, r, step)
            for s in sample:
                part.append( (s[0], s[1], s[-1]) )
            # cleanup angles
            if FIX_ANGLES:
                for i, xy in enumerate(part[:-1]):
                    dx = xy[0] - part[i+1][0]
                    dy = xy[1] - part[i+1][1]

                    phi = np.arctan2(dy, dx)
                    path.append( (xy[0], xy[1], phi) ) 
                path.append(wp2)
            else:
                path = path + part
        elif model == Vehicle.BEZIER:
            control_point1 = wp1[0] + np.sin(wp1[2])*r, wp1[1] + np.cos(wp1[2])*r
            control_point2 = wp2[0] - np.sin(wp2[2])*r, wp2[1] - np.cos(wp2[2])*r
            nodes = np.asfortranarray([
                [wp1[0], control_point1[0], control_point2[0], wp2[0]],
                [wp1[1], control_point1[1], control_point2[1], wp2[1]]
            ])
            curve = bezier.Curve(nodes, degree=3)
            l = np.linspace(0.0, 1.0, num=int(curve.length / step))
            points = curve.evaluate_multi(l)
            angles = [curve.evaluate_hodograph(i) for i in l]
            angles = [np.arctan2(x[1], x[0])[0] for x in angles]
            for i, (x, y) in enumerate(points.transpose()):
                path.append( (x, y, angles[i]) )
            
                
            
            
        else:
            print("NO VEHICLE MODEL!")
    
        
    if waypoints[-1] != path[-1]:
        path.append(waypoints[-1])
    return path


def single_agent_objectives(agents, index, r=1, step=0.1, model=None, obstacles=None, metric=None):
    paths = [waypoints_to_path(agent, r=r, step=step, model=model) for agent in agents]
    robustness = single_robustness_from_paths(paths, i, obstacles=obstacles, metric=metric)
    lp = len(paths[index])
    return robustness, lp
    

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

def single_min_dist_per_agent(configuration,index):
    if configuration[index] is None:
        return np.inf
    lc = len(configuration[index])
    m = np.full((lc), np.inf)

    for j in range(lc):
        if configuration[j] is None:
            continue
        if j == index:
            continue
        d = (configuration[index][0] - configuration[j][0])**2
        d += (configuration[index][1] - configuration[j][1])**2
        m[j] = d
    return np.sqrt(np.min([m], axis=0))


def single_robustness_from_paths(paths, index, obstacles=None, metric=None):
    d_obstacles = []
    collision = False
    collision_value = 0
    agent = paths[index]
    # checking obstacle cost
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
    # if no collision check inter agent paths
    d_agents = np.array(d_obstacles) * 2
    for configuration in itertools.zip_longest(*paths):
        d_agents = np.min([d_agents, single_min_dist_per_agent(configuration, index)], axis=0)
    if metric is None or metric is Metric.MIN:
        return np.min(d_agents)
    if metric is Metric.MIXED:
        return np.min(d_agents) + np.mean(d_agents) / 1000
    if metric is Metric.MEAN:
        return np.mean(d_agents)
    print("unkown metric")
    return None
    
    
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
    
