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
        'n_gens': 250,
        'population_size': 64,
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
        "runs" : 11,
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
    """
    add_jobs_for_each_model(settings.copy(), **job_settings.copy())
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

    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "sigma"
    for a in np.logspace(-2.5, -1.5, num=5):
        s["sigma"] = a
        j["experiment"] = f"sigma_{a:.3f}"
        add_jobs_for_each_model(s.copy(), **j.copy())

    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "cx"
    for a in np.linspace(0.4, 1.0, num=4):
        for b in np.linspace(0.4, 1.0, num=4):
            s["cxpb"] = a
            s["mutpb"] = b
            j["experiment"] = f"cx_{a:.2f}_{b:.2f}"
            add_jobs_for_each_model(s.copy(), **j.copy())


    """
    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "mut"
    s["n_gens"] = 300
    for a in np.linspace(0.0, 1.0, num=3):
        for b in np.linspace(0.0, 1.0, num=3):
            for c in np.linspace(0.0, 1.0, num=3):
                if a == 0.0 and b == 0.0 and c == 0.0:
                    continue
                s["mutation_p"] = (0.25, a, b, c)
                j["experiment"] = f"mut_{a:.2f}_{b:.2f}_{c:.2f}"
                add_jobs_for_each_model(s.copy(), **j.copy())
                time.sleep(2)
    """
