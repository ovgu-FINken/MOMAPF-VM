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
        'domain': (0, 200.0),
        'n_agents': 5,
        'n_waypoints': 3,
        'n_gens': 500,
        'population_size': 4*25,
        'cxpb': 0.75,
        'mutpb': 1.0,
        'mutation_p': (1.0, 1.0, 1.0),
        'sigma' : 0.1,
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
        "experiment" : "dubins_baseline",
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
    s["population_size"] = 16
    add_jobs_to_db(s, **j)
    add_jobs_to_db(settings.copy(), **job_settings.copy())

    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "dubins_cx"
    for a in np.linspace(0.0, 1.0, num=5):
        for b in np.linspace(0.0, 1.0, num=5):
            s["cxpb"] = a
            s["mutpb"] = b
            j["experiment"] = f"dubins_cx_{a:.2f}_{b:.2f}"
            add_jobs_to_db(s.copy(), **j.copy())
            time.sleep(2)

    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "dubins_mut"
    s["n_gens"] = 300
    for a in np.linspace(0.0, 1.0, num=5):
        for b in np.linspace(0.0, 1.0, num=5):
            for c in np.linspace(0.0, 1.0, num=5):
                if a == 0.0 and b == 0.0 and c == 0.0:
                    continue
                s["mutation_p"] = (a, b, c)
                j["experiment"] = f"dubins_mut_{a:.2f}_{b:.2f}_{c:.2f}"
                add_jobs_to_db(s.copy(), **j.copy())
                time.sleep(2)

    s = settings.copy()
    s["model"] = Vehicle.STRAIGHT
    j = job_settings.copy()
    j["group"] = "straight_cx"
    for a in np.linspace(0.0, 1.0, num=5):
        for b in np.linspace(0.0, 1.0, num=5):
            s["cxpb"] = a
            s["mutpb"] = b
            j["experiment"] = f"straight_cx_{a:.2f}_{b:.2f}"
            add_jobs_to_db(s.copy(), **j.copy())
            time.sleep(2)

    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "straight_mut"
    s["model"] = Vehicle.STRAIGHT
    s["n_gens"] = 300
    for a in np.linspace(0.0, 1.0, num=5):
        for b in np.linspace(0.0, 1.0, num=5):
            for c in np.linspace(0.0, 1.0, num=5):
                if a == 0.0 and b == 0.0 and c == 0.0:
                    continue
                s["mutation_p"] = (a, b, c)
                j["experiment"] = f"straight_mut_{a:.2f}_{b:.2f}_{c:.2f}"
                add_jobs_to_db(s.copy(), **j.copy())
                time.sleep(2)
