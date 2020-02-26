#!/usr/bin/env python
import numpy as np

import cProfile
import pstats
import random
import sqlalchemy
import git
import json
import traceback
import warnings
import os
import time
from multiprocessing import Pool, TimeoutError
import multiprocessing
import logging

from enum import IntEnum
from deap import base, creator, tools, algorithms

from problem import *
from obstacle_map import *




class JobStatus(IntEnum):
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
        
    def pop_to_df(self, population):
        data = []
        fronts = tools.sortNondominated(population, len(population))
        for i, front in enumerate(fronts):
            for j, ind in enumerate(front):
                f = self.problem.evaluate(ind)
                feasible = "feasible"
                if f[0] > self.settings['feasiblity_threshold']:
                    feasible = "infeasible"
                i_data = {
                    'front' : i,
                    'non_dominated': i == 0,
                    'crowding_distance': np.min([2, ind.fitness.crowding_dist]),
                    'individual': j,
                    'value': json.dumps(ind),
                    'robustness' : f[0],
                    'flowtime' : f[1],
                    'makespan' : f[2],
                    'collision' : feasible,
                }
                data.append(i_data)
        return pd.DataFrame(data)
    
    def hypervolume2(self, df, objective_1="robustness", objective_2="flowtime"):
        ref = self.settings["hv_ref"]
        x_last = ref[0]
        hv = 0
        for i, x in df.loc[df.non_dominated].sort_values(by=[objective_1], ascending=False).iterrows():
            if x[objective_1] > ref[0]:
                continue
            if x[objective_2] > ref[1]:
                continue
            delta_f1 = x_last - x[objective_1]
            delta_f2 = ref[1] - x[objective_2]
            hv += delta_f1 * delta_f2
            x_last = x[objective_1]
        return hv
    
    def _hv_pop(self, pop):
        return self.hypervolume2(self.pop_to_df(pop))

    def run(self, verbose=False):
        # for compatibility
        settings = self.settings
        toolbox = self.toolbox
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("median", np.median, axis=0)
        # stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "max", "median", "hv"

        pop = toolbox.population(n=settings['population_size'])
        pop = toolbox.select(pop, settings['population_size']) # for computing crowding distance

        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        record = stats.compile(pop)
        record["hv"] = self._hv_pop(pop)
        logbook.record(gen=0, evals=len(pop), **record)
        if verbose:
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
            record["hv"] = self._hv_pop(pop)
            logbook.record(gen=g, evals=evals, **record)
            if verbose:
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
            "hv": log["hv"]
        }
        for i, _ in enumerate(log['median']):
            data_i[f"f_{i}_median"] = log['median'][i]
            data_i[f"f_{i}_min"] = log['min'][i]
            data_i[f"f_{i}_max"] = log['max'][i]
        data.append(data_i)
    return pd.DataFrame(data)
  
    
def execute_job(job=None):
    if job is None:
        print("no job provided")
        return
    start_time = time.time()
    experiment = Experiment(job['settings'])
    experiment.setup()
    experiment.seed(job['seed'])
    pop, log = experiment.run()
    pop = experiment.pop_to_df(pop)
    job_time = time.time() - start_time
    return pop, log, job_time


def get_key(filename="db.key"):
    s = None
    with open(filename) as f:
        s = f.read()[:-1]
    return s   


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
        job['settings'] = json.loads(job['settings'])
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
        return self.execute_and_save(job)

    def save_results(self, res, job):
        pop, log, job_time = res
        self.save_population(pop, job=job)
        self.save_logbook(job=job, logbook=log)
        self.set_job_status(job, status=JobStatus.DONE, time=job_time)

    def execute_and_save(self, job):
        # TODO: check if job is taken by other process after updating table
        print(f"excuting job {job['index']}")
        self.set_job_status(job, status=JobStatus.IN_PROGRESS)
        try:
            res = execute_job(job)
            # save results
            save_results(res, job)
        except:
            self.set_job_status(job, status=JobStatus.FAILED)
            raise
        return True
    
    def execute_pool(self, workers=2):
        with Pool(processes=workers) as pool:
            job = self.fetch_job()
            jobs = {}
            handles = {}
            try:
                while job is not None or len(handles.keys()) > 0:
                    if job is not None:
                        self.set_job_status(job, status=JobStatus.IN_PROGRESS)
                        jobs[job['index']] = job
                        handles[job['index']] = pool.apply_async(execute_job, (job,))
                        print(f"starting job {job['index']}.")
                    #handles.append(pool.apply_async(time.sleep, (3,)))
                    #print(handles)
                    while len(handles.keys()) >= workers:
                        time.sleep(5)
                        completed = []
                        for k, v in handles.items():
                            if v.ready():
                                completed.append(k)
                                if v.successful():
                                    self.save_results(v.get(), jobs[k])
                                    print(f"job {k} successful")
                                else:
                                    self.set_job_status(jobs[k], status=JobStatus.FAILED)
                                    print(f"job {k} failed")
                        for k in completed:
                            del handles[k]
                            del jobs[k]
                    time.sleep(1)
                    job = self.fetch_job()
            except:
                for k, v in jobs.items():
                    self.set_job_status(v, status=JobStatus.FAILED)
                raise
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
        df['group'] = job['group']
        df['run'] = job['run']
        df.to_sql("logbooks", self.db, if_exists="append")
            

        
    
    def save_population(self, df, job=None):
        if job is None:
            print("not saving population, job is None")
            return
        
        df['job_seed'] = job['seed']
        df['job_index'] = job['index']
        df['experiment'] = job['experiment']
        df['group'] = job['group']
        df['run'] = job['run']
        df.to_sql("populations", self.db, if_exists="append")
        
        
        
def add_jobs_to_db(settings, db=None, experiment=None, group=None, time=-1, pid=-1, user="default", runs=31, delete=False):
    """Add new jobs (runs) to the experiment db with the given settings."""
    assert(experiment is not None)
    assert(db is not None)
    if group is None:
        group = "default"
    jobs = [{
        "experiment" : experiment,
        "run" : i,
        "seed": i+1000,
        "status": JobStatus.TODO.value,
        "commit": get_commit(),
        "user": user,
        "time": time,
        "pid": pid,
        "group" : group,
        "settings": json.dumps(settings)
    } for i in range(runs)]
    df_jobs = pd.DataFrame(jobs)
    if delete:
        df_jobs.to_sql("jobs", con=db, if_exists="replace")
    else:
        old_jobs = pd.read_sql("jobs", con=db)
        df_jobs.index = range(len(old_jobs), len(old_jobs) + len(df_jobs))
        df_jobs.to_sql("jobs", con=db, if_exists="append")
        

def get_names(db):
    df_pop = pd.read_sql("populations", con=db)
    print(df_pop["experiment"].unique())
    
def jobs(db):
    df_jobs = pd.read_sql_table("jobs", con=db)
    return df_jobs
    
    
def read_experiment(db, name=None, verbose=False):
    df_pop = pd.read_sql("populations", con=db)
    df_stats = pd.read_sql("logbooks", con=db)
        
    if name is not None:
        df_pop, df_stats = df_pop.loc[df_pop['experiment']==name], df_stats.loc[df_stats['experiment']==name]
    
    if verbose:
        data = []
        for exp in df_pop["experiment"].unique():
            ji = df_pop.loc[df_pop["experiment"] == exp]["job_index"].values[0]
            settings = fetch_settings(jobs(db), job_index=ji)
            settings["mutp_0"] = settings["mutation_p"][0]
            settings["mutp_1"] = settings["mutation_p"][1]
            settings["mutp_2"] = settings["mutation_p"][2]
            settings["experiment"] = exp
            data.append(settings)
        df = pd.DataFrame(data)
        df_pop = df_pop.join(df.set_index("experiment"), on="experiment")
        df_stats = df_stats.join(df.set_index("experiment"), on="experiment")
    return df_pop, df_stats

def fetch_settings(df_jobs, job_index=None):
    assert(job_index is not None)
    row = df_jobs.loc[df_jobs.index == job_index]
    s = row.iloc[0]
    return json.loads(s["settings"])

def plot_indivdual(row, df_jobs=None, plot=True, animation=False, animation_file=None):
    """creates a plot from the individual in resulting dataframe"""
    settings = fetch_settings(df_jobs, job_index=row['job_index'])
    ex = Experiment(settings)
    ex.setup()
    ind = json.loads(row['value'])
    if plot:
        ex.problem.solution_plot(ind)
    if animation:
        ex.problem.solution_animation(ind, filename=animation_file)
    return settings, ex
    
    
    
    
if __name__ == "__main__":
    mpl = multiprocessing.log_to_stderr()
    mpl.setLevel(logging.INFO)
    engine = sqlalchemy.create_engine(get_key(filename="db.key"))
    runner = ExperimentRunner(engine)
    runner.execute_pool(workers=70)
    
