import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
import yaml

from path import *


def random_waypoint(domain=(0.0, 100.0), velocity_control=False):
    x = np.random.uniform(low=domain[0], high=domain[1])
    y = np.random.uniform(low=domain[0], high=domain[1])
    phi = np.random.uniform(low=0.0, high=np.pi*2)
    if velocity_control:
        return [x, y, phi, np.random.uniform(0.2, 1.0)]
    return [x, y, phi]

def circle_waypoint(domain=(0.0, 100.0), r=75, angle=0):
    center = (domain[0] + domain[1]) / 2
    x = np.cos(angle) * r + center
    y = np.sin(angle) * r + center
    return [x, y, angle+np.pi/2]

def line_configuration(n_agents=1, domain=(0.0, 200.0)):
    dh= (domain[1] - domain[0]) / (n_agents + 1)
    c = (domain[1] - domain[0]) / 2
    start = []
    end = []
    
    if n_agents % 2 == 0:
        oy = dh / 2
    else:
        oy = 0
    ox = 70
    sx = -1
    sy = -1
    for i in range(n_agents):
        start.append( (c + ox * sx , c - oy * sy, 0 ) )
        end.append( (c - ox * sx, c + oy * sy, np.pi ) )
        if (i + n_agents) % 2 == 1:
            oy += dh
            sx = -1 * sx
        sy = -1 * sy
    return start, end

def polynomial_mut(x, sigma=0.1, p=0.5, lower=0.0, upper=1.0, **_):
    r = sigma
    if np.random.uniform() < 0.5:
        return x
    a = 0
    u = np.random.uniform()
    x = np.clip(x, lower, upper)
    if u <= 0.5:
        xl = (x - lower) / (upper - lower)
        a = (2.0 * u + ( 1.0 - 2.0*u )*(1.0-xl)**(r+1))**(1.0/(r+1.0))-1.0 
    else:
        xr = (upper - x) / (upper - lower)
        a = 1.0 - (2 * (1 - u) + (2 * u - 0.5) * (1 - xr)**(r+1))**(1/r+1)
    return np.clip(x + a, lower, upper)

def gauss_mut(x, sigma=0.1, lower=0.0, upper=1.0, **_):
    u = np.random.normal(0, sigma * (upper - lower))
    return np.clip(x + u, lower, upper)


def start_from_params(pose_x=0.0, pose_y=0.0, pose_theta=0.0, **_):
    return pose_x, pose_y, pose_theta
    
def read_robots_file(robots_file, n):
    with open(robots_file, 'r') as stream:
        robots = yaml.safe_load(stream)
    
    starts = []
    goals = []
    for robot in robots[:n]:
        starts.append(start_from_params(**robot))
        goals.append( (robot['goals']['x'][0], robot['goals']['y'][0], robot['goals']['theta'][0] ) )
    return starts, goals
    
class DubinsMOMAPF():

    def __init__(self, n_agents=4, domain=(0.0, 100.00), radius=5.0, step=0.1, model=Vehicle.DUBINS, obstacles=None, metric=None, velocity_control=False, configuration="line",robots_yaml=None, **unused_settings):
        if configuration == "line":
            self.start, self.goals = line_configuration(n_agents=n_agents, domain=domain)
        elif configuration == "circle":
            self.start = [circle_waypoint(domain=domain, r=70, angle=i) for i in np.linspace(0, 2*np.pi, n_agents+1)[:-1]]
            self.goals= [circle_waypoint(domain=domain, r=70, angle=i+np.pi) for i in np.linspace(0, 2*np.pi, n_agents+1)[:-1]]
        elif configuration == "yaml":
            self.start, self.goals = read_robots_file(robots_yaml, n_agents)
            self.start = [ obstacles.metric_to_px_coordinates(*p) for p in self.start ]
            self.goals = [ obstacles.metric_to_px_coordinates(*p) for p in self.goals ]
        else:
            print("configuration not found")
        self.r = radius
        self.step = step
        self.n_agents = n_agents
        self.model = model
        self._last_fig = None
        self._anim_paths = None
        self.obstacles=obstacles
        self.metric = metric
        self.velocity_control = velocity_control
        self.domain = (0.0, np.max(obstacles.size))
        
    
    def waypoints_to_path(self, wps):
        return waypoints_to_path(wps, model=self.model, r=self.r, step=self.step, FIX_ANGLES=True)

    
    def agents_objectives(self, agents):
        return agents_objectives(agents, r=self.r, step=self.step, model=self.model, obstacles=self.obstacles, metric=self.metric)

    def decode(self, vector):
        """
        decode an encoded solution to waypoints
        wp with or without velocity
        - x
        - y
        - theta
        - (velocity)
        """
        wps = []
        for i, wp in enumerate(self.start):
            wps.append([list(wp)])
        if self.velocity_control:
            for i, wp in enumerate(zip(vector[0::4], vector[1::4], vector[2::4], vector[3::4])):
                wps[int(i * 4 / len(vector) * self.n_agents)].append(list(wp))
        else:
            for i, wp in enumerate(zip(vector[0::3], vector[1::3], vector[2::3])):
                wps[int(i * 3 / len(vector) * self.n_agents)].append(list(wp))
            
        for i, wp in enumerate(self.goals):
            wps[i].append(list(wp))
        return wps
    
    def encode(self, agents):
        """
        encode a solution from waypionts
        - ditch start and goal wp
        - [(agent=0, wp=0), (agent=0, wp=1) ... (agent=0, wp=n), (agent=1, wp=0), (agent=1, wp=1), ... (wp=n, agent=k)]
        """
        vector = []
        wps = [wp[1:-1] for wp in agents]
        for i, wp in enumerate(wps):
            while len(wp) > 0:
                vector = vector + list(wp[0])
                wp = wp[1:]
        return vector

    def evaluate(self, vector, pop = None, k=10, novelty_only=False):
        agents = None
        robustness, time,  length = 0,0,0
        if not novelty_only:
            agents = self.decode(vector)
            robustness, time, length = self.agents_objectives(agents)
        nearest = []
        knn = None
        v = np.array(vector)
        novelty_minimisation = 0.0
        if pop is not None:
            k = np.min([k, len(pop)])
            for ind in pop:
                d = np.linalg.norm(v - np.array(ind))
                nearest.append(d)
            knn = sorted(nearest)[:k]
            s = np.sum(knn)
            if s == 0.0:
                novelty_minimisation = 1e20
            else:
                novelty_minimisation = 1.0 / s
        return 100-robustness, time, length, novelty_minimisation
    
    def all_mutations(self, vector, type_distribution={'skip': 1.0, 'full': 1.0}, mode="gauss", **params):
        x = np.random.rand() * sum(type_distribution.values())
        a = 0
        
        
        a += type_distribution['skip']
        if x < a:
            return self.skip_mutation(vector, mode=mode, params=params)
        
        a += type_distribution['uniform']
        if x < a:
            return self.uniform_mutation(vector)
        
        a += type_distribution['waypoint']
        if x < a:
            return self.mutate(vector, params=params, mode=mode)
            
        a += type_distribution['agent']
        if x < a:
            return self.mutate_agent(vector, params=params, mode=mode)
        
        return self.mutate_full(vector, params=params, mode=mode)
    
    def mutate(self, vector, params={}, mode="gauss", debug=False, **_):
        mutation_function = gauss_mut
        params['sigma'] = params['sigma_waypoint']
        if mode == "polynomial":
            mutation_function = polynomial_mut
        wps = self.decode(vector)
        agent = np.random.randint(len(wps))
        wp = 1 + np.random.randint(len(wps[agent])-2)
        if debug:
            print(f"changing wp {wp} of agent {agent}, with sigma={params['sigma']}")
            print(f"old: {wps[agent][wp]}")
        wps[agent][wp][0] = mutation_function(wps[agent][wp][0], lower=self.domain[0], upper=self.domain[1], **params)
        wps[agent][wp][1] = mutation_function(wps[agent][wp][1], lower=self.domain[0], upper=self.domain[1], **params)
        wps[agent][wp][2] = mutation_function(wps[agent][wp][2]%(2*np.pi), lower=0.0, upper=2.0*np.pi, **params)
        if len(wps[agent][wp])>3:
            wps[agent][wp][3] = mutation_function(wps[agent][wp][3], lower=0.2, upper=1.0, **params)
        result = self.encode(wps)
        if debug:
            print(f"new: {wps[agent][wp]}")
            print(f"full: {result}")
        #assert(result != vector)
        for i, v in enumerate(result):
            vector[i] = v
        return vector,
    
    def mutate_agent(self, vector, params={}, mode="gauss"):
        params['sigma'] = params['sigma_agent']
        mutation_function = gauss_mut
        if mode == "polynomial":
            mutation_function = polynomial_mut
        wps = self.decode(vector).copy()
        agent = np.random.randint(len(wps))
        for wp in range(1,len(wps[agent])-1):
            wps[agent][wp][0] = mutation_function(wps[agent][wp][0], lower=self.domain[0], upper=self.domain[1], **params)
            wps[agent][wp][1] = mutation_function(wps[agent][wp][1], lower=self.domain[0], upper=self.domain[1], **params)
            wps[agent][wp][2] = mutation_function(wps[agent][wp][2], lower=0.0, upper=2.0*np.pi, **params)
            if len(wps[agent][wp])>3:
                wps[agent][wp][3] = mutation_function(wps[agent][wp][3], lower=0.2, upper=1.0, **params)
        result = self.encode(wps)
        #assert(result != vector)
        for i, v in enumerate(result):
            vector[i] = v
        return vector,
    
    def mutate_full(self, vector, params={}, mode="gauss"):
        params['sigma'] = params['sigma_full']
        mutation_function = gauss_mut
        if mode == "polynomial":
            mutation_function = polynomial_mut
        wps = self.decode(vector).copy()
        for agent in range(len(wps)):
            for wp in range(1,len(wps[agent])-1):
                wps[agent][wp][0] = mutation_function(wps[agent][wp][0], lower=self.domain[0], upper=self.domain[1], **params)
                wps[agent][wp][1] = mutation_function(wps[agent][wp][1], lower=self.domain[0], upper=self.domain[1], **params)
                wps[agent][wp][2] = mutation_function(wps[agent][wp][2], lower=0.0, upper=2.0*np.pi, **params)
                if len(wps[agent][wp])>3:
                    wps[agent][wp][3] = mutation_function(wps[agent][wp][3], lower=0.2, upper=1.0, **params)
        result = self.encode(wps)
        #assert(result != vector)
        for i, v in enumerate(result):
            vector[i] = v
        return vector,
    
    def uniform_mutation(self, vector, debug=False):
        agent_i = np.random.randint(self.n_agents)
        agents = self.decode(vector)
        if debug:
            print(f"changing agent {agent_i}")
            print(agents[agent_i])
        agents[agent_i] = [random_waypoint(domain=self.domain, velocity_control=self.velocity_control) for _ in agents[0]]
        vector[:] = self.encode(agents)
        if debug:
            print(agents[agent_i])
            print(vector)
        return vector,

    def skip_mutation(self, vector, debug=False, **kwargs):
        agent_i = np.random.randint(self.n_agents)
        wps = self.decode(vector)
        i = 1+np.random.randint(len(wps[agent_i])-2)
        before = wps[agent_i][i-1]
        after = wps[agent_i][i+1]
        path = self.waypoints_to_path([before, after])
        if len(path) < 10:
            if debug:
                print("skip-mutation-skipped")
            return self.mutate(vector, **kwargs)
        ix_rand = np.random.randint(low=1, high=len(path)-2)
        wp = list(path[ix_rand])
        if len(wps[agent_i][i])>3:
            wp.append(wps[agent_i][i][3])
        if debug:
            print(f"adapding WP{i} of A{agent_i}")
            print(wp)
        wps[agent_i][i] = wp
        vector[:] = self.encode(wps)
        return vector,
    
    def crossover(self, a, b):
        agent_i = np.random.randint(self.n_agents)
        l_agent = len(a) / self.n_agents
        ai0 = int(agent_i * l_agent)
        ai1 = int((agent_i + 1) * l_agent)
        a[ai0:ai1], b[ai0: ai1] = b[ai0:ai1], a[ai0:ai1]
        return a, b

    def evaluate_weighted_sum(self, vector, w_r=0, w_m=0, w_f=10, c_r = 20):
        agents = self.decode(vector)
        robustness, makespan, flowtime = self.agents_objectives(agents)
        return w_r * (100 - np.min([c_r,robustness])) + w_m * makespan + w_f * flowtime,

    def _get_point(self, center, radius, orin):
        x = center[0] + radius * np.cos(orin)
        y = center[1] + radius * np.sin(orin)
        return (x,y)
    
    def animation_update(self, i):
        x = []
        y = []
        for j, path in enumerate(self._anim_paths):
            if len(path) > i:
                self.circles[j].center = (path[i][0], path[i][1])
                self.polygons[j].xy = [self._get_point( (path[i][0],path[i][1]), self.circles[j].radius*0.5, path[i][2]+offset) for offset in [0, -np.pi*2/3, np.pi*2/3]]
            else:
                self.circles[j].radius = 0
                self.polygons[j].xy = [self._get_point( (path[-1][0],path[-1][1]), self.circles[j].radius*0.05, path[-1][2]+offset) for offset in [0, -np.pi*2/3, np.pi*2/3]]

    def agents_animation(self, agents, filename=None, plot_range=None, safety_radius=10, duration = 5.0, speedup=3):
        plt.ioff()
        fig = plt.figure(figsize=(5,5))
        objectives = self.agents_objectives(agents)
        for agent in agents:
            plot_waypoints(agent, alpha=0.5)
        if self.obstacles is not None:
            self.obstacles.heatmap(plot_range=plot_range)
        self._anim_paths = None
        #step = objectives[1]/100
        step = self.step
        self._anim_paths = [waypoints_to_path(agent, r=self.r, step=step, model=self.model, FIX_ANGLES=True) for agent in agents]
        self.circles = [plt.Circle( (a[0][0], a[0][1]), safety_radius, fc=(0.3, 0.1, 0.1, 0.1), ec=(0.8,0.1,0.1)) for a in agents]
        self.polygons = [plt.Polygon( [self._get_point( (a[0][0],a[0][1]),safety_radius*0.5, a[0][2]+offset) for offset in [0, -np.pi*2/3, np.pi*2/3]], fc=(0.8,0.1,0.1))for a in agents]
        
        for path in self._anim_paths:
            plot_waypoints(path),
        ax = plt.gca()
        for patch in self.circles:
            ax.add_patch(patch)
        for poly in self.polygons:
            ax.add_patch(poly)
        #plt.title(f"robustness: {objectives[0]}\nmakespan: {objectives[1]}\nflowtime: {objectives[2]}")
        plt.tight_layout()
        longest = max([len(p) for p in self._anim_paths])
        intervall = duration / longest * 1000
        anim = animation.FuncAnimation(fig, self.animation_update, frames=range(0, longest, speedup),
                                           interval=intervall, blit=False, repeat_delay=1000)

        if filename is not None:
            anim.save(filename)
        display(anim)
        plt.close(fig)
        plt.ion()
    
    def solution_animation(self, ind, **kwargs):
        self.agents_animation(self.decode(ind), **kwargs)
    
    def solution_plot(self, ind, plot_range=None, legend=False, show=True):
        wps = self.decode(ind)
        df_wp = wps_to_df(wps)
        df_paths = wps_to_df([waypoints_to_path(wp, r=self.r, step=self.step, model=self.model) for wp in wps])
        palette = sns.color_palette("muted", n_colors=self.n_agents)
        if show:
            plt.figure()
        if self.obstacles is not None:
            self.obstacles.heatmap(plot_range=plot_range)
        sns.scatterplot(data=df_wp, x="x", y="y", hue="agent", marker="x", palette=palette, legend=legend)
        sns.lineplot(data=df_paths, x="x", y="y", hue="agent", palette=palette, sort=False, legend=False)
        if show:
            plt.show()
    
    def path_df(self, ind, additional_args={}):
        wps = self.decode(ind)
        df = wps_to_df([waypoints_to_path(wp, r=self.r, step=self.step, model=self.model) for wp in wps])
        for k, v in additional_args.items():
            df[k] = v
        return df