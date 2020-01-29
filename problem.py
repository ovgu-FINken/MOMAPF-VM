import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

class DubinsMOMAPF():

    def __init__(self, n_agents=4, domain=(0.0, 100.00), radius=5.0, step=0.1, model=DUBINS, obstacles=None,**unused_settings):
        print(unused_settings)
        self.start = [circle_waypoint(domain=domain, r=60, angle=2*np.pi*(0.1 + 0.8 * i/n_agents)) for i in range(n_agents)]
        self.goals = [circle_waypoint(domain=domain, r=80, angle=-2*np.pi*(0.1 + 0.8 * i/n_agents)) for i in range(n_agents)]
        self.r = radius
        self.step = step
        self.n_agents = n_agents
        self.domain = domain
        self.model = model
        self._last_fig = None
        self._anim_paths = None
        self.sct = None
        self.obstacles=obstacles
        
    
    def waypoints_to_path(self, wps):
        return waypoints_to_path(wps, model=self.model, r=self.r, step=self.step)

    
    def agents_objectives(self, agents):
        return agents_objectives(agents, r=self.r, step=self.step, model=self.model, obstacles=self.obstacles)

    def decode(self, vector):
        wps = []
        for i, wp in enumerate(self.start):
            wps.append([wp])
        for i, wp in enumerate(zip(vector[0::3], vector[1::3], vector[2::3])):
            wps[i%self.n_agents].append(wp)
        for i, wp in enumerate(self.goals):
            wps[i].append(wp)
        return wps
    
    def encode(self, agents):
        vector = []
        wps = [wp[1:-1] for wp in agents]
        while len(wps[-1]) > 0:
            for i, wp in enumerate(wps):
                vector = vector + list(wp[0])
                wps[i] = wp[1:]
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
        return self.mutate(vector, sigma=sigma)

    def mutate(self, vector, sigma=0.1):
        i = np.random.randint(len(vector) / 3)
        s = sigma*(self.domain[1]-self.domain[0])
        vector[3*i+0] += np.random.normal(0.0, s)
        vector[3*i+1] += np.random.normal(0.0, s)
        vector[3*i+2] += np.random.normal(0.0, sigma*2*np.pi)
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
        i = np.random.randint(len(vector) / self.n_agents / 3) + 1
        wps = self.decode(vector)
        before = wps[agent_i][i-1]
        after = wps[agent_i][i+1]
        path = self.waypoints_to_path([before, after])
        if len(path) < 10:
            if debug:
                print("skip-mutation-skipped")
            return self.mutate(vector)
        ix_rand = np.random.randint(low=1, high=len(path)-2)
        if debug:
            print(f"adapding WP{i} of A{agent_i}")
        wp = path[ix_rand]
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

    
    def animation_update(self, i):
        x = []
        y = []
        for path in self._anim_paths:
            if len(path) > i:
                x.append(path[i][0])
                y.append(path[i][1])
        self.sct.set_data(x, y)
        return self.sct,

    def agents_animation(self, agents, filename=None, plot_range=None):
        plt.ioff()
        fig = plt.figure()
        objectives = self.agents_objectives(agents)
        for agent in agents:
            plot_waypoints(agent, alpha=0.5)
        if self.obstacles is not None:
            if plot_range is None:
                plot_range = np.arange(*self.domain)
            self.obstacles.heatmap(plot_range=plot_range)
        self._anim_paths = None
        self.sct = None
        self._anim_paths = [waypoints_to_path(agent, r=self.r, step=objectives[1]/100, model=self.model) for agent in agents]
        for path in self._anim_paths:
            plot_waypoints(path)
        self.sct, = plt.plot([], [], "ro")
        plt.title(f"robustness: {objectives[0]}\nmakespan: {objectives[1]}\nflowtime: {objectives[2]}")
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
    
    def solution_plot(self, ind, plot_range=None):
        wps = self.decode(ind)
        df_wp = wps_to_df(wps)
        df_paths = wps_to_df([waypoints_to_path(wp, r=self.r, step=self.step, model=self.model) for wp in wps])
        palette = sns.color_palette("muted", n_colors=self.n_agents)
        plt.figure()
        if self.obstacles is not None:
            if plot_range is None:
                plot_range = np.arange(*self.domain)
            self.obstacles.heatmap(plot_range=plot_range)
        sns.scatterplot(data=df_wp, x="x", y="y", hue="agent", marker="x", palette=palette)
        sns.lineplot(data=df_paths, x="x", y="y", hue="agent", palette=palette, sort=False, legend=False)
        plt.show()