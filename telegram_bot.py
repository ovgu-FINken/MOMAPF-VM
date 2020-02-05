import sqlalchemy
import pandas as pd
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
from experiment import *
from time import sleep


engine = sqlalchemy.create_engine('sqlite:///experiments.db')
notify_chat_ids = set()


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


def notify(update, context):
    chat_id = update.message.chat_id
    notify_chat_ids.add(chat_id)
    print(f"register {chat_id} for notification")
    contextd.boxt.send_message(chat_id=update.effective_chat.id, text="Hello, registered for updates.")

def get_token(filename="telegram.key"):
    s = None
    with open(filename) as f:
        s = f.read()[:-1]
    return s

def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

def status(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=job_status_msg())

updater = Updater(token=get_token(), use_context=True)
dispatcher = updater.dispatcher

echo_handler = MessageHandler(Filters.text, echo)
dispatcher.add_handler(echo_handler)
status_handler = CommandHandler('status', status)
dispatcher.add_handler(status_handler)
notify_handler = CommandHandler('notify', notify)
dispatcher.add_handler(notify_handler)

if __name__ == "__main__":
    print(job_status_msg())
    updater.start_polling()
    nearly_done = False
    done = False
    while True:
        df = jobs()
        msg = None
        if not nearly_done:
            if JobStatus.TODO.value not in df["status"].values:
                nearly_done = True
                msg = "Nearly done, no jobs queued"
        if nearly_done and not done:
            if JobStatus.TODO.value not in df["status"].values:
                done = True
                msg = "Done, all jobs completed or failed"
        if msg is not None:
            print(msg)
            for chat in notify_chat_ids:
                updater.bot.send_message(chat, msg)

