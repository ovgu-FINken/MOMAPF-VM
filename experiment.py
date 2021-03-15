#!/usr/bin/env python
import numpy as np

import cProfile
import pstats
import random
import sqlalchemy
try:
    import git
except ImportError:
    git = None
    print("could not import git module")
import json
import traceback
import warnings
import os
import time
from multiprocessing import Pool, TimeoutError
import multiprocessing
import subprocess
import logging
import itertools
import sys

from enum import IntEnum
from deap import base, creator, tools, algorithms

from problem import *
from obstacle_map import *

import argparse

try:
    # try importing the C version
    from deap.tools._hypervolume import hv as hv
except ImportError:
    # fallback on python version
    from deap.tools._hypervolume import pyhv as hv


def hypervolume2(ref, df, objective_1="robustness", objective_2="time"):
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

def hypervolume(ref, df, toolbox, objectives=("robustness", "time", "length")):
    pointset = []
    for _, row in df.loc[df.feasible].iterrows():
        objective_values = tuple( (row[o] for o in objectives) )
        pointset.append(objective_values)
    if len(pointset) == 0:
        return 0
    return hv.hypervolume(pointset, ref)

def pdom(a, b, ndim=None):
    """return True if a dominates b (minimization) """
    aLTb = False
    if ndim is None:
        ndim = min(len(a), len(b))
    for fa, fb in zip(a[:ndim], b[:ndim]):
        if not fa <= fb :
            return False
        elif fa < fb:
            aLTb = True
    return aLTb
    
class JobStatus(IntEnum):
    TODO = 0
    IN_PROGRESS = 1
    DONE = 2
    FAILED = 3
    RESERVED = 4
    
class Experiment:
    def __init__(self, settings):
        self.settings = settings
    
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(random.randint(0, 2**32-1))
        
    def setup(self):
        obstacles = ObstacleMap(filename=self.settings['map_name'])
        self.problem = DubinsMOMAPF(obstacles=obstacles, **self.settings)
        
        # deap setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.settings["use_novelty"]:
                creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
            else:
                creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
                
            creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_pos", np.random.uniform, self.settings['domain'][0], self.settings['domain'][1])
        self.toolbox.register("attr_angle", np.random.uniform, 0, 2*np.pi)
        if self.settings['velocity_control']:
            self.toolbox.register("attr_velocity", np.random.uniform, 0.5, 1.0)
            self.toolbox.register("individual", tools.initCycle, creator.Individual,
                                  (self.toolbox.attr_pos, self.toolbox.attr_pos, self.toolbox.attr_angle, self.toolbox.attr_velocity),
                                  n=self.settings['n_agents'] * self.settings['n_waypoints']
                                 )
        else:
            self.toolbox.register("individual", tools.initCycle, creator.Individual,
                                  (self.toolbox.attr_pos, self.toolbox.attr_pos, self.toolbox.attr_angle),
                                  n=self.settings['n_agents'] * self.settings['n_waypoints']
                                 )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        #self.toolbox.register("evaluate", problem.evaluate_weighted_sum)
        self.toolbox.register("evaluate", self.problem.evaluate, k=self.settings["novelty_k"])
        self.toolbox.register("mate", self.problem.crossover)
        self.toolbox.register("mutate", self.problem.all_mutations, **self.settings["mutation_settings"])
        self.toolbox.register("select", tools.selNSGA2)
        
    def pop_to_df(self, population, evaluate=True):
        data = []
        for i, ind in enumerate(population):
            f = None
            if evaluate:
                f = self.problem.evaluate(ind, pop=population)
            else:
                f = ind.fitness.values[0:3]
            i_data = {
                'individual': i,
                'robustness' : f[0],
                'time' : f[1],
                'length' : f[2],
                'feasible' : f[0] < self.settings['feasiblity_threshold'],
            }
            try:
                i_data['value'] = json.dumps(ind),
            except:
                print(ind)
                raise
            data.append(i_data)
        return pd.DataFrame(data)
    
    
    def _hv_df(self, df):
        return hypervolume(self.settings["hv_ref"], df, self.toolbox)

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
        logbook.header = "gen", "evals", "min", "max", "median", "hv", "walltime", "len(archive)"

        pop = toolbox.population(n=settings['population_size'])
        pop = toolbox.select(pop, settings['population_size']) # for computing crowding distance
        archive = []
        ndim = 3

        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
            
            #archive insertion
            dominated = False
            for a in archive:
                if pdom(ind.fitness.values, a.fitness.values, ndim=ndim):
                    archive.remove(a)
                elif pdom(a.fitness.values, ind.fitness.values, ndim=ndim):
                    dominated = True
                    break
            if not dominated:
                archive += [ind]

        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind, pop=archive)
            
        record = stats.compile(pop)
        if len(archive) > 0:
            record["hv"] = self._hv_df(self.pop_to_df(archive, evaluate=False))
        else:
            record["hv"] = 0
        record["walltime"] = 0
        record["len(archive)"] = len(archive)
        logbook.record(gen=0, evals=len(pop), **record)
        if verbose:
            print(logbook.stream)
        start_time = time.time()
        
        for g in range(1, settings['n_gens']):
            # Select and clone the next generation individuals
            #offspring = toolbox.clone(pop)
            offspring = tools.selTournamentDCD(pop, settings['population_size'])
            offspring = [toolbox.clone(ind) for ind in offspring]

            #offspring = algorithms.varAnd(offspring, toolbox, settings['cxpb'], settings['mutpb'])

            # Apply crossover and mutation on the offspring
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                change1, change2 = False, False
                if np.random.rand() < settings['cxpb']:
                    ind1, ind2 = toolbox.mate(ind1, ind2)
                    change1, change2 = True, True
                
                if not change1 or np.random.rand() < settings['mutpb']:
                    change1 = True
                    toolbox.mutate(ind1)
                if not change2 or np.random.rand() < settings['mutpb']:
                    change2 = True
                    toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
                assert(change1)
                assert(change2)
            
            
            #add new offspring to pop, that are not duplicates
            for ind in offspring:
                duplicate = False
                for x in pop:
                    if x == ind:
                        duplicate = True
                        break
                if not duplicate:
                    pop.append(ind)
            
            # fitness evaluation and archiving
            evals = 0
            for ind in pop:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind, pop=archive)
                    #archive insertion
                    dominated = False
                    for a in archive:
                        if pdom(ind.fitness.values, a.fitness.values, ndim=ndim):
                            archive.remove(a)
                        elif pdom(a.fitness.values, ind.fitness.values, ndim=ndim):
                            dominated = True
                            break
                    if not dominated:
                        archive += [ind]
                    evals += 1
                else:
                    ind.fitness.values = ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2], toolbox.evaluate(ind, pop=pop, novelty_only=True)[3]
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
            if g % 10 == 0 or verbose:
                record = stats.compile(pop)
                if len(archive) > 0:
                    record["hv"] = self._hv_df(self.pop_to_df(archive, evaluate=False))
                else:
                    record["hv"] = 0
                record["walltime"] = time.time() - start_time
                record["len(archive)"] = len(archive)
                logbook.record(gen=g, evals=evals, **record)
                if verbose:
                    print(logbook.stream)

        for ind in archive:
            ind.fitness.values = toolbox.evaluate(ind,pop=pop)
        #recompute crowding distance
        archive = toolbox.select(archive, len(archive))
        return archive, logbook
    
def get_commit():
    if git is None:
        return ""
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
            "hv": log["hv"],
            "walltime": log["walltime"],
            "archive": log["len(archive)"]
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
    
    def reset_running_jobs(self):
        """ set job status of all jobs with status RUNNING or RESERVED to FAILED """
        select = sqlalchemy.sql.select([self.table_jobs]).where( (self.table_jobs.c.status == JobStatus.IN_PROGRESS.value) | (self.table_jobs.c.status == JobStatus.RESERVED.value))
            
        r = self.db.execute(select)
        rows = r.fetchall()
        r.close()
        for row in rows:
            job = {}
            for col in self.table_jobs.columns.keys():
                job[str(col)] = row[col]
            self.set_job_status(job=job, status=JobStatus.FAILED)
    
    def fetch_job(self, verbose=False, job_index:int=None, reserve=False):
        select = None
        if job_index is None:
            select = sqlalchemy.sql.select([self.table_jobs]).where( (self.table_jobs.c.status == JobStatus.TODO.value) | (self.table_jobs.c.status == JobStatus.FAILED.value))
        else:
            select = sqlalchemy.sql.select([self.table_jobs]).where( (self.table_jobs.c.index == job_index) )
            
        r = self.db.execute(select)
        row = r.fetchone()
        r.close()
        if row is None:
            if verbose:
                print("could not fetch job")
            return
        if row[self.table_jobs.c.commit] != get_commit():
            if verbose:
                print("WARNING: commits do not match")
        job = {}
        for col in self.table_jobs.columns.keys():
            job[str(col)] = row[col]
        job['settings'] = json.loads(job['settings'])
        if reserve:
            print(f"reserve: {job['index']}")
            self.set_job_status(job=job, status=JobStatus.RESERVED)
            time.sleep(0.8)
            db_job = self.fetch_job(verbose=verbose, job_index=job['index'], reserve=False)
            if db_job['pid'] == os.getpid():
                if verbose:
                    print(f"fetched job: {job}")
                return job
            else:
                time.sleep(3*np.random.rand())
                return self.fetch_job(verbose=verbose, reserve=True, job_index=job_index)
        if verbose:
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
        
    
    def fetch_and_execute(self, job_index=None):
        """fetch jobs with the TODO-status from the DB and run the experiment.
        
        1. Fetch Job
        2. Set Job-status to running
        3. Execute Experiment
        4. a) Save Results and set job-status to DONE
        4. b) In case of exception set job-status to FAIL
        
        """
        reserve = (job_index is None)
        job = self.fetch_job(job_index=job_index, reserve=reserve)
        if job is None:
            return False
        try:
            print(f"executing: {job['index']}")
            return self.execute_and_save(job)
        except Exception as ex:
            print(ex)
            self.set_job_status(job=job, status=JobStatus.FAILED)
            sys.exit(1)
        
    def save_results(self, res, job):
        pop, log, job_time = res
        self.save_population(pop, job=job)
        self.save_logbook(job=job, logbook=log)
        self.set_job_status(job, status=JobStatus.DONE, time=job_time)

    def execute_and_save(self, job):
        print(f"excuting job {job['index']}")
        self.set_job_status(job, status=JobStatus.IN_PROGRESS)
        try:
            res = execute_job(job)
            # save results
            self.save_results(res, job)
        except:
            self.set_job_status(job, status=JobStatus.FAILED)
            sys.exit(1)
        return True
    
    def execute_pool(self, workers=2):
        with Pool(processes=workers) as pool:
            jobs = {}
            handles = {}
            print("entering worker loop")
            try:
                while True:
                    job = self.fetch_job(reserve=True)
                    if job is not None and len(handles.keys()) < workers:
                        self.set_job_status(job, status=JobStatus.IN_PROGRESS)
                        jobs[job['index']] = job
                        handles[job['index']] = pool.apply_async(execute_job, (job,))
                        print(f"{time.strftime('%H:%M:%S')} -- start {job['index']}.")
                    if len(handles.keys()) == workers:
                        time.sleep(5)
                    completed = []
                    for k, v in handles.items():
                        if v.ready():
                            completed.append(k)
                            if v.successful():
                                self.save_results(v.get(), jobs[k])
                                print(f"{time.strftime('%H:%M:%S')} -- job {k} successful")
                            else:
                                self.set_job_status(jobs[k], status=JobStatus.FAILED)
                                print(f"{time.strftime('%H:%M:%S')} -- job {k} failed")
                                try:
                                    print(v.get())
                                except:
                                    pass


                    for k in completed:
                        del handles[k]
                        del jobs[k]
                    time.sleep(0.5)
            except:
                for k, v in jobs.items():
                    self.set_job_status(v, status=JobStatus.FAILED)
                sys.exit(1)
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
        
        
def add_jobs_for_each_model(settings, experiment=None, group=None, **job_settings):
    assert(experiment is not None)
    if group is None:
        group = ""
    for vm in Vehicle:
        settings["model"] = vm
        add_jobs_to_db(settings, experiment = f"{vm.name}_{experiment}", group = f"{group}", **job_settings)
        
def add_jobs_to_db(settings, db=None, experiment=None, group=None, time=-1, pid=-1, user="default", runs=31, delete=False, seed_offset=1000):
    """Add new jobs (runs) to the experiment db with the given settings."""
    assert(experiment is not None)
    assert(db is not None)
    if group is None:
        group = "default"
    jobs = [{
        "experiment" : experiment,
        "run" : i,
        "seed": i+seed_offset,
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
        sql = 'DROP TABLE IF EXISTS populations;'
        db.execute(sql)
        sql = 'DROP TABLE IF EXISTS logbooks;'
        db.execute(sql)
        df_jobs.to_sql("jobs", con=db, if_exists="replace")
    else:
        old_jobs = pd.read_sql("jobs", con=db)
        min_index = old_jobs.index.max() + 1
        df_jobs.index = range(min_index, min_index + len(df_jobs))
        df_jobs.to_sql("jobs", con=db, if_exists="append")
        

def get_names(db):
    df_pop = pd.read_sql("populations", con=db)
    print(df_pop["experiment"].unique())
    
def jobs(db):
    df_jobs = pd.read_sql_table("jobs", con=db)
    return df_jobs
    
    
def compute_combined_front(df, objectives=("robustness", "time", "length"), colname=None, groups=None, experiments=None):
    if colname is None:
        colname = "combined_front"
    if groups is None:
        groups = df["group"].unique()
    if experiments is None:
        experiments = df["experiment"].unique()
    df.loc[df.group.isin(groups)&df.experiment.isin(experiments),colname] = False
    view = df.loc[df.group.isin(groups)&df.experiment.isin(experiments)]
    
    l = []
    for i, ind in view.iterrows():
        l.append((i, tuple([ind[o] for o in objectives])))
    
    print(len(l))
    a = set()
    for i in l:
        non_dom = True
        remove = set()
        for j in a:
            if pdom(i[1], j[1]):
                remove.add(j)
                continue
            if pdom(j[1], i[1]):
                non_dom = False
                break
        a -= remove
        if non_dom:
            a.add(i)
    print(len(a))
    
    colnr = df.columns.get_loc(colname)
    df.iloc[[i[0] for i in a], colnr] = True
        
    return df


def read_table(table, experiment=None, con=None):
    if experiment is None:
        return pd.read_sql(table, con=con)
    metadata = sqlalchemy.MetaData(con)
    t = sqlalchemy.Table(table, metadata, autoload=True)
    sel = sqlalchemy.select([t]).where(t.c.experiment == experiment)
    return pd.read_sql_query(sel, con=con)
    

def read_experiment(db, experiment=None, verbose=False):
    df_pop = read_table("populations", con=db, experiment=experiment)
    print("finished reading populations")
    df_stats = read_table("logbooks", con=db, experiment=experiment)
    print("finished reading logbooks")
    
    
    if verbose:
        data = []
        df_jobs = read_table("jobs", con=db, experiment=experiment)
        for exp in df_pop["experiment"].unique():
            ji = df_pop.loc[df_pop["experiment"] == exp, "job_index"].values[0]
            settings = fetch_settings(df_jobs, job_index=ji)
            settings["experiment"] = exp
            data.append(settings)
        df = pd.DataFrame(data)
        df_pop = df_pop.join(df.set_index("experiment"), on="experiment")
        df_stats = df_stats.join(df.set_index("experiment"), on="experiment")
        #for group in df_pop["group"].unique():
        #    compute_combined_front(df_pop, colname="group_front", groups=[group])
        print("finished reading settings")
        for exp in df_pop["experiment"].unique():
            compute_combined_front(df_pop,colname="experiment_front", experiments=[exp])
            print(f"finished computing front for {exp}")
    return df_pop, df_stats

def fetch_settings(df_jobs, job_index=None):
    assert(job_index is not None)
    s = df_jobs.loc[df_jobs["index"] == job_index, "settings"].values[0]
    return json.loads(s)

def plot_indivdual(row, df_jobs=None, plot=True, animation=False, animation_file=None, show=True):
    """creates a plot from the individual in resulting dataframe"""
    settings = fetch_settings(df_jobs, job_index=row['job_index'])
    ex = Experiment(settings)
    ex.setup()
    ind = json.loads(row['value'])
    if plot:
        ex.problem.solution_plot(ind, show=show)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        plt.title(f"$f_R^*={row['robustness']:.1f}$, $f_L={row['flowtime']:.1f}$")
        plt.tight_layout()
    if animation:
        ex.problem.solution_animation(ind, filename=animation_file)
    return settings, ex
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Runner')
    parser.add_argument("--db", type=str)
    parser.add_argument("--multiprocessing", type=int)
    parser.add_argument("--run", type=int, nargs='+')
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    
    key = "db.key"
    if args.db:
        key = args.db
        
    engine = sqlalchemy.create_engine(get_key(filename=key))
    runner = ExperimentRunner(engine)
    if args.run:
        print(f"executing runs: {args.run}")
        for i in args.run:
            runner.fetch_and_execute(job_index=i)
        
    elif args.fetch:
        time.sleep(3 * np.random.rand())
        runner.fetch_and_execute()
        print("... done.")
        if args.loop:
            while True:
                runner.fetch_and_execute()
                print("... done.")
    
    elif args.multiprocessing:
        print("multiprocessing.pool")
        mpl = multiprocessing.log_to_stderr()
        mpl.setLevel(logging.WARN)
        runner.execute_pool(workers=args.multiprocessing)
        
    elif args.slurm:
        for i in range(100):
            job = runner.fetch_job(reserve=True)
            subprocess.Popen(f"srun -n 1 ./job.bash --run {job['index']}".split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    elif args.reset:
        runner.reset_running_jobs()
        
    
