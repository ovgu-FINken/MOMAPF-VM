#!/usr/bin/env python
import numpy as np

import cProfile
import pstats
import random
import sqlalchemy
import git
import pickle
import traceback
import warnings
import os
import time

from enum import Enum
from deap import base, creator, tools, algorithms

from problem import *
from obstacle_map import *

class JobStatus(Enum):
    TODO = 0
    IN_PROGRESS = 1
    DONE = 2
    FAILED = 3
    
class Experiment:
    def __init__(self, settings):
        self.settings = settings
    
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(random.randint(0, 2**32-1))
        
    def setup(self):
        obstacles = ObstacleMap(filename=self.settings['map_name'])
        self.problem = DubinsMOMAPF(**self.settings, obstacles=obstacles)
        
        # deap setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_pos", np.random.uniform, self.settings['domain'][0], self.settings['domain'][1])
        self.toolbox.register("attr_angle", np.random.uniform, 0, 2*np.pi)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.attr_pos, self.toolbox.attr_pos, self.toolbox.attr_angle),
                              n=self.settings['n_agents'] * self.settings['n_waypoints']
                             )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        #self.toolbox.register("evaluate", problem.evaluate_weighted_sum)
        self.toolbox.register("evaluate", self.problem.evaluate)
        self.toolbox.register("mate", self.problem.crossover)
        self.toolbox.register("mutate", self.problem.all_mutations, p=self.settings['mutation_p'], sigma=self.settings['sigma'])
        self.toolbox.register("select", tools.selNSGA2)
    
    def run(self):
        # for compatibility
        settings = self.settings
        toolbox = self.toolbox
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("median", np.median, axis=0)
        # stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = toolbox.population(n=settings['population_size'])
        pop = toolbox.select(pop, settings['population_size']) # for computing crowding distance

        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)
        print(logbook.stream)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
            
        for g in range(1, settings['n_gens']):
            # Select and clone the next generation individuals
            #offspring = toolbox.clone(pop)
            offspring = tools.selTournamentDCD(pop, settings['population_size'])
            offspring = [toolbox.clone(ind) for ind in offspring]

            #offspring = algorithms.varAnd(offspring, toolbox, settings['cxpb'], settings['mutpb'])

            # Apply crossover and mutation on the offspring
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() <= settings['cxpb']:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            pop = pop + offspring
            evals = 0
            for ind in pop:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
                    evals += 1
            pop_feasible = []
            pop_infeasible = []
            for ind in pop:
                if ind.fitness.values[0] < settings['feasiblity_threshold']:
                    pop_feasible.append(ind)
                else:
                    pop_infeasible.append(ind)
            if len(pop_feasible) < settings['population_size']:
                pop_feasible = pop_feasible + tools.selBest(pop_infeasible, settings['population_size'] - len(pop))
            # even needs to be done, if infeasible solutions exist, because crowding distance needs to be computed
            pop = toolbox.select(pop_feasible, settings['population_size'])
            record = stats.compile(pop)
            logbook.record(gen=g, evals=evals, **record)
            print(logbook.stream)

        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        return pop, logbook
    
    
def get_commit():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha

def logbook_to_df(logbook):
    data = []
    evals = 0
    for log in logbook:
        evals += log['evals']
        data_i = {
            "generation": log['gen'],
            "evals": evals,
        }
        for i, _ in enumerate(log['median']):
            data_i[f"f_{i}_median"] = log['median'][i]
            data_i[f"f_{i}_min"] = log['min'][i]
            data_i[f"f_{i}_max"] = log['max'][i]
        data.append(data_i)
    return pd.DataFrame(data)
    
    
class ExperimentRunner:
    def __init__(self, db):
        assert db.has_table("jobs")
        #assert db.hastable("populations")
        #assert db.hastable("logs")
        self.db = db
        self.metadata = sqlalchemy.MetaData(self.db)
        self.table_jobs = sqlalchemy.Table('jobs', self.metadata, autoload=True)
        #self.table_populations = sqlalchemy.Table('populations', self.metadata, autoload=True)
        #self.table_logs = sqlalchemy.Table('logs', self.metadata, autoload=True)
    
    def fetch_job(self):
        select = sqlalchemy.sql.select([self.table_jobs]).where(self.table_jobs.c.status == JobStatus.TODO.value)
        r = self.db.execute(select)
        row = r.fetchone()
        r.close()
        if row is None:
            return
        if row[self.table_jobs.c.commit] != get_commit():
            print("WARNING: commits do not match")
        job = {}
        for col in self.table_jobs.columns.keys():
            job[str(col)] = row[col]
        job['settings'] = pickle.loads(job['settings'])
        print(f"fetched job: {job}")
        return job

    def set_job_status(self, job=None, status=None, time=0):
        assert(status is not None)
        if job is None:
            print("trying to set job None to active")
            return
        update = self.table_jobs.update()\
            .where(self.table_jobs.c.index == job['index'])\
            .values(status=status.value, pid=os.getpid(), time=int(time))
        self.db.execute(update).close()
        
    def execute_job(self, job=None):
        if job is None:
            print("no job provided")
            return
        self.experiment = Experiment(job['settings'])
        self.experiment.setup()
        self.experiment.seed(job['seed'])
        result = self.experiment.run()
        return result
    
    def fetch_and_execute(self):
        """fetch jobs with the TODO-status from the DB and run the experiment.
        
        1. Fetch Job
        2. Set Job-status to running
        3. Execute Experiment
        4. a) Save Results and set job-status to DONE
        4. b) In case of exception set job-status to FAIL
        
        """
        job = self.fetch_job()
        if job is None:
            return False
        # TODO: check if job is taken by other process after updating table
        self.set_job_status(job, status=JobStatus.IN_PROGRESS)
        try:
            start_time = time.time()
            pop, log = self.execute_job(job)
            job_time = time.time() - start_time
            # save results
            self.save_population(job=job, population=pop)
            self.save_logbook(job=job, logbook=log)
        except Exception as e:
            print("experiment failed, resetting job status")
            print(e)
            traceback.print_last()
            self.set_job_status(job, status=JobStatus.FAILED)
            raise e
        # TODO: save time in job table
        self.set_job_status(job, status=JobStatus.DONE, time=job_time)
        self.experiment = None
        return True
    
    def save_logbook(self, logbook=None, job=None):
        if logbook is None:
            print("not saving logbook None")
            return
        if job is None:
            print("not saving logbook, job is None")
            return
        df = logbook_to_df(logbook)
        df['job_seed'] = job['seed']
        df['job_index'] = job['index']
        df['experiment'] = job['experiment']
        df['run'] = job['run']
        df.to_sql("logbooks", self.db, if_exists="append")
            
    
    def save_population(self, population=None, job=None):
        if population is None:
            print("not saving population None")
            return
        if job is None:
            print("not saving population, job is None")
            return
        data = []
        
        fronts = tools.sortNondominated(population, len(population))
        for i, front in enumerate(fronts):
            for j, ind in enumerate(front):
                f = self.experiment.problem.evaluate(ind)
                feasible = "feasible"
                if f[0] > job['settings']['feasiblity_threshold']:
                    feasible = "infeasible"
                i_data = {
                    'front' : i,
                    'non_dominated': i == 0,
                    'crowding_distance': np.min([2, ind.fitness.crowding_dist]),
                    'individual': j,
                    'value': pickle.dumps(ind),
                    'robustness' : f[0],
                    'flowtime' : f[1],
                    'makespan' : f[2],
                    'collision' : feasible,
                    'job_seed' : job['seed'],
                    'job_index' : job['index'],
                    'experiment' : job['experiment'],
                    'run' : job['run'],
                }
                data.append(i_data)
        df = pd.DataFrame(data)
        df.to_sql("populations", self.db, if_exists="append")
        
        
def add_jobs_to_db(settings, db=None, name=None, time=-1, pid=-1, user="default", runs=31, delete=False):
    """Add new jobs (runs) to the experiment db with the given settings."""
    assert(name is not None)
    assert(db is not None)
    df_jobs = None
    if not delete:
        df_jobs = pd.read_sql_table("jobs", con=db, index_col="index")
    jobs = [{
        "experiment" : name,
        "run" : i,
        "seed": i+1000,
        "status": JobStatus.TODO.value,
        "commit": get_commit(),
        "user": user,
        "time": time,
        "pid": pid,
        "settings": pickle.dumps(settings)
    } for i in range(runs)]
    if delete:
        df_jobs = pd.DataFrame(jobs)
    else:
        df_jobs = df_jobs.append(jobs, ignore_index=True)
    df_jobs.to_sql("jobs", con=db, if_exists="replace") 
    
if __name__ == "__main__":
    engine = sqlalchemy.create_engine('sqlite:///experiments.db')
    runner = ExperimentRunner(engine)
    running = True
    while running:
        running = runner.fetch_and_execute()
    
