#!/usr/bin/env python
import sqlalchemy
import numpy as np
import pandas as pd
import seaborn as sns
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import *
from time import sleep
import io
import logging
import argparse

engine = sqlalchemy.create_engine(get_key(filename="db.key"))
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def job_status_msg():
    df = jobs(engine)
    s = ""
    for group in df["group"].unique():
        s += f"group {group}\n"
        for value in df.loc[df["group"]==group]["status"].unique():
            vc = df.loc[df["group"] == group]["status"].value_counts()
            status = JobStatus(value)
            s += f"{status.name}: {vc[value]}\n"
        s += "\n"
    return s 


class TBot:
    def __init__(self, keyfile):
        self.done = False
        self.nearly_done = False
        self.notify_chat_ids = set()
        self.handlers = {}
        self.updater = Updater(token=get_key(filename=keyfile), use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.parse_error_chat_id = None

        self.add_handler(name="status", function=self.status)
        self.add_handler(name="notify", function=self.notify)
        self.add_handler(name="unsubscribe", function=self.unsubscribe)
        self.add_handler(name="help", function=self.commands)
        self.add_handler(name="commands", function=self.commands)
        self.add_handler(name="start", function=self.commands)
        self.add_handler(name="plot", function=self.scatterplot)
        self.add_handler(name="convergence", function=self.convergence_plot)
        self.add_handler(name="test", function=self.test)

        self.echo_handler = MessageHandler(Filters.text, self.echo)
        self.dispatcher.add_handler(self.echo_handler)

        
    def add_handler(self, name=None, function=None):
        self.handlers[name] = CommandHandler(name, function)
        self.dispatcher.add_handler(self.handlers[name])
        
    def notify(self, update, context):
        chat_id = update.message.chat_id
        self.notify_chat_ids.add(chat_id)
        print(f"register {chat_id} for notification")
        context.bot.send_message(chat_id=update.effective_chat.id, text="Hello, registered for updates.")

    def unsubscribe(self, update, context):
        chat_id = update.message.chat_id
        self.notify_chat_ids.remove(chat_id)
        print(f"deregister {chat_id} for notification")
        context.bot.send_message(chat_id=update.effective_chat.id, text="Succesfully unsubscribed.")

    def echo(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

    def status(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=job_status_msg())
        
    def test(self, update, context):
        print("------------")
        parser = argparse.ArgumentParser('Test argument parsing')
        parser.add_argument("-x", nargs=1, default="robustness", choices=["robustness", "flowtime", "makespan"])
        parser.add_argument("-y", nargs=1, default="flowtime", choices=["robustness", "flowtime", "makespan"])
        x = parser.parse_args(context.args)
        print(str(x))
        context.bot.send_message(chat_id=update.effective_chat.id, text=str(x))
        print("------------")
        
    def commands(self, update, context):
        c = ["/" + x for x in self.handlers.keys()]
        msg = "Available Commands:\n"
        msg += "\n".join(c)
        context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    
    def loop(self):
        df = jobs(engine)
        msg = None
        if not self.nearly_done:
            if JobStatus.TODO.value not in df["status"].values:
                self.nearly_done = True
                msg = "Nearly self.done, no jobs queued"
        if self.nearly_done and not self.done:
            if JobStatus.TODO.value not in df["status"].values:
                self.done = True
                msg = "done"
        if msg is None and not self.done:
            msg = job_status_msg()
        else:
            return
        for chat in self.notify_chat_ids:
            self.updater.bot.send_message(chat, msg)
            
            
    def parse_error_message(self, status=0, message=None):
        self.parse_error = True
        self.updater.bot.send_message(self.parse_error_chat_id, message)
            
    def scatterplot(self, update, context):
        df_pop, _ = read_experiment(engine, verbose=True)
        parser = argparse.ArgumentParser('Plot argument parsing')
        columns = list(df_pop.keys())
        experiments = list(df_pop["experiment"].unique())
        groups = list(df_pop["group"].unique())
        parser.add_argument("-x", default="robustness", choices=columns)
        parser.add_argument("-y", default="flowtime", choices=columns)
        parser.add_argument("-hue", default="experiment", choices=columns)
        parser.add_argument("-row", default=None, choices=columns)
        parser.add_argument("-col", default=None, choices=columns)
        parser.add_argument("-style", default=None, choices=columns)
        parser.add_argument("-experiment", choices=experiments)
        parser.add_argument("-group", choices=groups)
        self.parse_error = False
        self.parse_error_chat_id = update.effective_chat.id
        parser.exit = self.parse_error_message
        plot_args = parser.parse_args(context.args)
        if self.parse_error:
            s = parser.format_usage()
            print(s)
            context.bot.send_message(chat_id=update.effective_chat.id, text=s)
            return
        
        context.bot.send_message(chat_id=update.effective_chat.id, text=str(plot_args))
        if plot_args.group is not None:
            df_pop = df_pop.loc[df_pop["group"] == plot_args.group]
        
        if plot_args.experiment is not None:
            df_pop = df_pop.loc[df_pop["experiment"] == plot_args.experiment]
        with plt.xkcd():
            rp = sns.relplot(data=df_pop.loc[df_pop["non_dominated"]],
                            x=plot_args.x,
                            y=plot_args.y,
                            hue=plot_args.hue,
                            style=plot_args.style,
                            row=plot_args.row,
                            col=plot_args.col,
                            alpha=.25
                            )#, palette="jet")
            
            #plt.tight_layout()
            buffer = io.BytesIO()
            rp.fig.savefig(buffer, format='png')

            buffer.seek(0)
            print("plotting ...")
            #plt.show()
            context.bot.send_photo(chat_id=update.effective_chat.id, photo=buffer, text="Plot")
    
    def convergence_plot(self, update, context):
        _, df_stats = read_experiment(engine, verbose=True)
        parser = argparse.ArgumentParser('Plot argument parsing')
        columns = list(df_stats.keys())
        experiments = list(df_stats["experiment"].unique())
        groups = list(df_stats["group"].unique())
        parser.add_argument("-x", default="generation", choices=columns)
        parser.add_argument("-y", default="hv", choices=columns)
        parser.add_argument("-hue", default="experiment", choices=columns)
        parser.add_argument("-row", default=None, choices=columns)
        parser.add_argument("-col", default=None, choices=columns)
        parser.add_argument("-style", default=None, choices=columns)
        parser.add_argument("-experiment", choices=experiments)
        parser.add_argument("-group", choices=groups)
        self.parse_error = False
        self.parse_error_chat_id = update.effective_chat.id
        parser.exit = self.parse_error_message
        plot_args = parser.parse_args(context.args)
        if self.parse_error:
            s = parser.format_usage()
            print(s)
            context.bot.send_message(chat_id=update.effective_chat.id, text=s)
            return
        
        context.bot.send_message(chat_id=update.effective_chat.id, text=str(plot_args))
        if plot_args.group is not None:
            df_stats = df_stats.loc[df_stats["group"] == plot_args.group]
        
        if plot_args.experiment is not None:
            df_stats = df_stats.loc[df_stats["experiment"] == plot_args.experiment]
        rp = sns.relplot(data=df_stats,
                        x=plot_args.x,
                        y=plot_args.y,
                        hue=plot_args.hue,
                        style=plot_args.style,
                        row=plot_args.row,
                        col=plot_args.col,
                        kind="line",
                        alpha=0.5,
                        ci=90,
                        estimator=np.median,
                        )#, palette="jet")

        #plt.tight_layout()
        buffer = io.BytesIO()
        rp.fig.savefig(buffer, format='png')

        buffer.seek(0)
        print("plotting ...")
        #plt.show()
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=buffer, text="Convergence Plot")

if __name__ == "__main__":
    print(job_status_msg())
    tbot = TBot("telegram.key")
    tbot.updater.start_polling()
    while True:
        tbot.loop()
        sleep(3600)

