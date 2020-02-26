
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
        'cxpb': 0.3,
        'mutpb': 1.0,
        'mutation_p': (1.0, 1.0, 1.0),
        'sigma' : 0.2,
        'model': Vehicle.DUBINS,
        'feasiblity_threshold': 95,
        'offset': (0, 0),
        'map_name': "cross.obstacles.npy",
        'metric': Metric.MIN,
        'hv_ref': (100, 400),
    }

    job_settings = {
        "delete" : True,
        "runs" : 31,
        "experiment" : "dubins_baseline",
        "group" : "default",
        "user" : "basti",
        "db" : engine,
    }
    add_jobs_to_db(settings, **job_settings)
    settings["n_gens"] = 200
    job_settings["delete"] = False

    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "dubins_mutation"
    for a in np.linspace(0.0, 2.0, num=5):
        for b in np.linspace(0.0, 2.0, num=5):
            for c in np.linspace(0.0, 2.0, num=5):
                if a == 0.0 and b == 0.0 and c == 0.0:
                    continue
                s["mutation_p"] = (a, b, c)
                j["experiment"] = f"dubins_mutation_{a:.1f}_{b:.1f}_{c:.1f}"
                add_jobs_to_db(s, **j)

    s = settings.copy()
    s["model"] = Vehicle.STRAIGHT
    j = job_settings.copy()
    j["group"] = "straight_mutation"
    for a in np.linspace(0.0, 2.0, num=5):
        for b in np.linspace(0.0, 2.0, num=5):
            for c in np.linspace(0.0, 2.0, num=5):
                if a == 0.0 and b == 0.0 and c == 0.0:
                    continue
                s["mutation_p"] = (a, b, c)
                j["experiment"] = f"straight_mutation_{a:.1f}_{b:.1f}_{c:.1f}"
                add_jobs_to_db(s, **j)


    s = settings.copy()
    j = job_settings.copy()
    j["group"] = "dubins_cx"
    for i in np.linspace(0.0, 1.0, num=11):
        s["cxpb"] = i
        j["experiment"] = f"dubins_cx_{i:.1f}"
        add_jobs_to_db(s, **j)

    s = settings.copy()
    j = job_settings.copy()
    s["model"] = Vehicle.STRAIGHT
    j["group"] = "straight_cx"
    for i in np.linspace(0.0, 1.0, num=11):
        s["cxpb"] = i
        j["experiment"] = f"straight_cx_{i:.1f}"
        add_jobs_to_db(s, **j)

