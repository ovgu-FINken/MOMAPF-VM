import sqlalchemy
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

def jobs():
    df_jobs = pd.read_sql_table("jobs", con=engine)
    return df_jobs

def job_status_msg():
    df = jobs()
    vc = df["status"].value_counts()
    s = ""
    for value in df["status"].unique():
        status = JobStatus(value)
        s += f"{status.name}: {vc[value]}\n"
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

        self.echo_handler = MessageHandler(Filters.text, self.echo)
        self.dispatcher.add_handler(self.echo_handler)
        self.add_handler(name="status", function=self.status)
        self.add_handler(name="notify", function=self.notify)
        self.add_handler(name="help", function=self.commands)
        self.add_handler(name="commands", function=self.commands)
        self.add_handler(name="start", function=self.commands)
        self.add_handler(name="plot", function=self.scatterplot)
        self.add_handler(name="test", function=self.test)

        
    def add_handler(self, name=None, function=None):
        self.handlers[name] = CommandHandler(name, function)
        self.dispatcher.add_handler(self.handlers[name])
        
    def notify(self, update, context):
        chat_id = update.message.chat_id
        self.notify_chat_ids.add(chat_id)
        print(f"register {chat_id} for notification")
        context.bot.send_message(chat_id=update.effective_chat.id, text="Hello, registered for updates.")


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
        df = jobs()
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
        print(msg)
        for chat in self.notify_chat_ids:
            self.updater.bot.send_message(chat, msg)
            
            
    def parse_error_message(self, status=0, message=None):
        self.parse_error = True
        self.updater.bot.send_message(self.parse_error_chat_id, message)
            
    def scatterplot(self, update, context):
        df_pop, _ = read_experiment(engine)
        parser = argparse.ArgumentParser('Plot argument parsing')
        columns = list(df_pop.keys())
        experiments = list(df_pop["experiment"].unique())
        parser.add_argument("-x", default="robustness", choices=columns)
        parser.add_argument("-y", default="flowtime", choices=columns)
        parser.add_argument("-hue", default="experiment", choices=columns)
        parser.add_argument("-experiment", choices=experiments)
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
        
        if plot_args.experiment is not None:
            df_pop = df_pop.loc[df_pop["experiment"] == plot_args.experiment]
        fig = plt.figure()
        sns.scatterplot(data=df_pop.loc[df_pop["non_dominated"]],
                        x=plot_args.x,
                        y=plot_args.y,
                        hue=plot_args.hue,
                        alpha=.5)#, palette="jet")
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')

        buffer.seek(0)
        print("potting ...")
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=buffer, text="Plot")
    

if __name__ == "__main__":
    print(job_status_msg())
    tbot = TBot("telegram.key")
    tbot.updater.start_polling()
    while True:
        tbot.loop()
        sleep(6000)

