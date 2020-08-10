import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

from path import *


def random_waypoint(domain=(0.0, 100.0)):
    x = np.random.uniform(low=domain[0], high=domain[1])
    y = np.random.uniform(low=domain[0], high=domain[1])
    phi = np.random.uniform(low=0.0, high=np.pi*2)
    return x, y, phi

def circle_waypoint(domain=(0.0, 100.0), r=75, angle=0):
    center = (domain[0] + domain[1]) / 2
    x = np.sin(angle) * r + center
    y = np.cos(angle) * r + center
    return (x, y, angle)

def line_configuration(n_agents=1, domain=(0.0, 200.0)):
    dh= (domain[1] - domain[0]) / (n_agents + 2)
    c = (domain[1] - domain[0]) / 2
    start = []
    end = []
    
    if n_agents % 2 == 0:
        oy = dh / 2
    else:
        oy = 0
    ox = 60
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

class DubinsMOMAPF():

    def __init__(self, n_agents=4, domain=(0.0, 100.00), radius=5.0, step=0.1, model=Vehicle.DUBINS, obstacles=None, metric=None, **unused_settings):
        self.start, self.goals = line_configuration(n_agents=n_agents, domain=domain)
        self.r = radius
        self.step = step
        self.n_agents = n_agents
        self.domain = domain
        self.model = model
        self._last_fig = None
        self._anim_paths = None
        self.sct = None
        self.obstacles=obstacles
        self.metric = metric
        
    
    def waypoints_to_path(self, wps):
        return waypoints_to_path(wps, model=self.model, r=self.r, step=self.step)

    
    def agents_objectives(self, agents):
        return agents_objectives(agents, r=self.r, step=self.step, model=self.model, obstacles=self.obstacles, metric=self.metric)

    def decode(self, vector):
        wps = []
        for i, wp in enumerate(self.start):
            wps.append([wp])
        for i, wp in enumerate(zip(vector[0::3], vector[1::3], vector[2::3])):
            wps[int(i * 3 / len(vector) * self.n_agents)].append(wp)
        for i, wp in enumerate(self.goals):
            wps[i].append(wp)
        return wps
    
    def encode(self, agents):
        vector = []
        wps = [wp[1:-1] for wp in agents]
        for i, wp in enumerate(wps):
            while len(wp) > 0:
                vector = vector + list(wp[0])
                wp = wp[1:]
        return vector

    def evaluate(self, vector):
        agents = self.decode(vector)
        robustness, makespan, flowtime = self.agents_objectives(agents)
        return 100-robustness, flowtime, makespan
    
    def all_mutations(self, vector, p=(1.0,1.0,1.0), sigma=None, **kwargs):
        x = np.random.rand() * np.sum(p)
        if x < p[0]:
            return self.skip_mutation(vector)
        if x < p[0] + p[1]:
            return self.uniform_mutation(vector)
        if len(p) > 3 and x >= p[0] + p[1] + p[2]:
            self.mutate_full(vector, sigma=sigma)
        return self.mutate(vector, sigma=sigma)

    def mutate(self, vector, sigma=0.1):
        i = np.random.randint(len(vector) / 3)
        s = sigma*(self.domain[1]-self.domain[0])
        vector[3*i+0] += np.random.normal(0.0, s)
        vector[3*i+1] += np.random.normal(0.0, s)
        vector[3*i+2] += np.random.normal(0.0, sigma*2*np.pi)
        return vector,
    
    def mutate_full(self, vector, sigma=0.01):
        s = sigma*(self.domain[1]-self.domain[0])
        for i, _ in enumerate(vector):
            if i%3 == 2:
                vector[i] += np.random.normal(0.0, sigma*2*np.pi)
            else:
                vector[i] += np.random.normal(0.0, s)
        return vector,
    
    def uniform_mutation(self, vector, debug=False):
        agent_i = np.random.randint(self.n_agents)
        agents = self.decode(vector)
        if debug:
            print(f"changing agent {agent_i}")
            print(agents[agent_i])
        agents[agent_i] = [random_waypoint(domain=self.domain) for _ in agents[0]]
        vector[:] = self.encode(agents)
        if debug:
            print(agents[agent_i])
            print(vector)
        return vector,

    def skip_mutation(self, vector, debug=False):
        agent_i = np.random.randint(self.n_agents)
        wps = self.decode(vector)
        i = np.random.randint(len(vector) / self.n_agents / 3) + 1
        before = wps[agent_i][i-1]
        after = wps[agent_i][i+1]
        path = self.waypoints_to_path([before, after])
        if len(path) < 10:
            if debug:
                print("skip-mutation-skipped")
            return self.mutate(vector)
        ix_rand = np.random.randint(low=1, high=len(path)-2)
        wp = path[ix_rand]
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
        #plt.plot(tri[:,0], tri[:,1], 'g-')
        for path in self._anim_paths:
            if len(path) > i:
                a = self._get_point(path[i], 10, path[i][2])
                b = self._get_point(path[i], 10/2, path[i][2]+150./180.*np.pi)
                c = self._get_point(path[i], 10/2, path[i][2]-150./180.*np.pi)
                x = x + [a[0], b[0], c[0]]
                y = y + [a[1], b[1], c[1]]
        self.sct.set_data(x, y)
        return self.sct,

    def agents_animation(self, agents, filename=None, plot_range=None):
        plt.ioff()
        fig = plt.figure(figsize=(5,5))
        objectives = self.agents_objectives(agents)
        for agent in agents:
            plot_waypoints(agent, alpha=0.5)
        if self.obstacles is not None:
            if plot_range is None:
                plot_range = np.arange(*self.domain)
            self.obstacles.heatmap(plot_range=plot_range)
        self._anim_paths = None
        self.sct = None
        self._anim_paths = [waypoints_to_path(agent, r=self.r, step=objectives[1]/100, model=self.model, FIX_ANGLES=True) for agent in agents]
        for path in self._anim_paths:
            plot_waypoints(path)
        self.sct, = plt.plot([], [], "ro")
        #plt.title(f"robustness: {objectives[0]}\nmakespan: {objectives[1]}\nflowtime: {objectives[2]}")
        plt.tight_layout()
        longest = max([len(p) for p in self._anim_paths])
        anim = animation.FuncAnimation(fig, self.animation_update, frames=longest,
                                           interval=100, blit=False, repeat_delay=1000)

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
            if plot_range is None:
                plot_range = np.arange(*self.domain)
            self.obstacles.heatmap(plot_range=plot_range)
        sns.scatterplot(data=df_wp, x="x", y="y", hue="agent", marker="x", palette=palette, legend=legend)
        sns.lineplot(data=df_paths, x="x", y="y", hue="agent", palette=palette, sort=False, legend=False)
        if show:
            plt.show()