#!/usr/bin/env python
import sqlalchemy
import pandas as pd
from path import *
from obstacle_map import *
from problem import *
from experiment import *

if __name__=="__main__":
    engine = sqlalchemy.create_engine(get_key(filename="db.key"))
    settings = {
        'radius': 10,
        'step': 1,
        'r_step': 0.2,
        'domain': (0, 200.0),
        'n_agents': 5,
        'n_waypoints': 3,
        'n_gens': 1000,
        'population_size': 100,
        'cxpb': 0.4,
        'mutpb': 0.8,
        'mutation_p': (0.5, .0, .0, 1.0),
        'sigma' : 0.01,
        'model': Vehicle.DUBINS,
        'feasiblity_threshold': 95,
        'offset': (0, 0),
        'map_name': "cross.obstacles.npy",
        'metric': Metric.MIN,
        'hv_ref': (100, 400),
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
    j["runs"] = 31
    j["delete"] = True
    s["n_gens"] = 50

    s["population_size"] = 16
    add_jobs_to_db(s, **j)
    #add_jobs_for_each_model(settings.copy(), **job_settings.copy())

    """
    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "pop"
    for a in [4*i*i for i in range(2,6)]:
        s["population_size"] = a
        s["n_gens"] = int(250 * 100 / a)
        print(f"{a} : {s['n_gens']}")
        j["experiment"] = f"pop_{a:.2f}"
        add_jobs_for_each_model(s.copy(), **j.copy())
    """
    """
    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "sigma"
    for a in np.logspace(-2.5, -1.5, num=5):
        s["sigma"] = a
        j["experiment"] = f"sigma_{a:.3f}"
        add_jobs_for_each_model(s.copy(), **j.copy())
    """
    
    labyrinth = [20, 40, 60, 80, 100, 120, 140, 160, 180]
    environments = [f"labyrinth_{i}.obstacles.npy" for i in labyrinth]
    environments += [f"labyrinth_2_{i}.obstacles.npy" for i in labyrinth]
    
    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "all"
    for a in [2,3,4,5,7,9,11]:
        for b in range(2, 6):
            for env in ["no.obstacles.npy", "cross.obstacles.npy", "bar.obstacles.npy"]+environemnts:
                s["n_agents"] = a
                s["n_waypoints"] = b
                s["map_name"] = env
                j["experiment"] = f"{env}_{a}_{b}"
                add_jobs_for_each_model(s.copy(), **j.copy())
