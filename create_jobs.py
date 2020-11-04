#!/usr/bin/env python
import sqlalchemy
import pandas as pd
from path import *
from obstacle_map import *
from problem import *
from experiment import *

if __name__=="__main__":
    engine = sqlalchemy.create_engine(get_key(filename="db.key"))
    mutation_settings = {
        'type_distribution' : {
                'skip' : 1.0,
                'uniform' : 0.0,
                'waypoint' : 1.0,
                'agent' : 1.0,
                'full' : 1.0,
            }, # distribution of mutation types
        'sigma_waypoint' : 0.3, # sigma for gauss-distribution in waypoint-gauss-mutation
        'sigma_agent' : 0.1, # sigma for gauss-distribution in agent-wide mutation
        'sigma_full' : 0.03, # sigma for gauss-mutation of all waypoints at the same time
        'mode' : "polynomial",
        'p' : 0.5,
    }
    settings = {
        'radius': 10, # turning radius (dubins vehicle)
        'model': Vehicle.DUBINS_ADAPTIVE, # vehicle model
        'step': 1, # step size for simulated behaviour
        'domain': (0, 200.0), # area of operation (-100, 100) means that the vehicles will move in a square from (x=-100, y=-100) to (x=100, y=100)
        'min_speed' : 0.2, 
        'n_agents': 7, # number of agents
        'n_waypoints': 3, # waypoints per agent (excluding start and end)
        'n_gens': 500, # number of generations to run the algorithm
        'population_size': 64, # population size for the algorithm, shoulod be divisible by 4 for nsga2
        'cxpb': .4, # crossover probablity
        'mutpb': 0.8, # mutation rate (not applicable in nsga2)
        'mutation_settings' : mutation_settings,

        'feasiblity_threshold': 99.0, # how robust a solution has to be to be regarded feasible (100-min_dist)
        'offset': (0, 0), # offset of the map to the agents
        'map_name': "obstacles/gaps_3_easy_60.npy", # name of the obstacle-map file
        'metric': Metric.MIN, # metric to use in fitness calculation
        'hv_ref': (100, 800, 400), # reference for hyper volume
        'velocity_control': True, # turn on velocity control (4th dimension on wp)
        'novelty_k': 5, # k for knn- in novelty objective
        'use_novelty': False,
        'configuration': 'line',
    }
    job_settings = {
        "delete" : False,
        "runs" : 31,
        "experiment" : "baseline",
        "group" : "default",
        "user" : "basti",
        "db" : engine,
    }

    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "quick"
    j["experiment"] = "quick"
    j["runs"] = 11
    j["delete"] = True
    s["n_gens"] = 50
    s["population_size"] = 32
    s["use_novelty"] = True
    add_jobs_to_db(s, **j)
    
    labyrinth = [95, 100, 105, 120]
    gaps = [2, 3]
    envs = []
    for i in labyrinth:
        for j in gaps:
            envs = envs + [f"obstacles/gaps_{j}_medium_{i}.npy"]

    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "circle"
    s['configuration'] = 'circle'
    s['map_name'] = 'obstacles/empty.npy'
    for a in [3, 5, 7]:
        for b in [2, 3, 5]:
            s["n_agents"] = a
            s["n_waypoints"] = b
            j["experiment"] = f"circle_{a}_{b}"
            add_jobs_for_each_model(s.copy(), **j.copy())
    
    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "normal"
    for a in [3,5,7]:
        for b in [2, 3,5]:
            for env in envs:
                s["n_agents"] = a
                s["n_waypoints"] = b
                s["map_name"] = env
                j["experiment"] = f"{env}_{a}_{b}"
                add_jobs_for_each_model(s.copy(), **j.copy())


    s = settings.copy()
    s['use_novelty'] = True
    j = job_settings.copy()
    j["group"] = "novelty"
    for a in [3,5,7]:
        for b in [2, 3,5]:
            for env in envs:
                s["n_agents"] = a
                s["n_waypoints"] = b
                s["map_name"] = env
                j["experiment"] = f"nov_{env}_{a}_{b}"
                add_jobs_for_each_model(s.copy(), **j.copy())

    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "nov_circle"
    s['use_novelty'] = True
    s['configuration'] = 'circle'
    s['map_name'] = 'obstacles/empty.npy'
    for a in [3, 5, 7]:
        for b in [2, 3, 5]:
            s["n_agents"] = a
            s["n_waypoints"] = b
            j["experiment"] = f"nov_circle_{a}_{b}"
            add_jobs_for_each_model(s.copy(), **j.copy())
