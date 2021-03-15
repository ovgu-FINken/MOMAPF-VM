
import sqlalchemy
import pandas as pd

from experiment import get_key

if __name__=="__main__":
    db = sqlalchemy.create_engine(get_key(filename="db.key"))
    sqlite = sqlalchemy.create_engine("sqlite:///sqlite_dump.db")
    
    gen = pd.read_sql("jobs", con=db, chunksize=1000)
    df = pd.concat(gen)
    print("read jobs done.")
    df.to_sql("jobs", con=sqlite, if_exists="replace", chunksize=1000)
    print("write jobs done.")
    del df
    gen = pd.read_sql("populations", con=db, chunksize=500)
    print("read populations done.")
    first = True
    for chunk in gen:
        if first:
            chunk.to_sql("populations", con=sqlite, if_exists="replace", chunksize=500)
            first = False
        else:
            chunk.to_sql("populations", con=sqlite, if_exists="append", chunksize=500)
    print("write populations done.")
    gen = pd.read_sql("logbooks", con=db, chunksize=1000)
    print("read logbooks done.")
    first = True
    for chunk in gen:
        if first:
            chunk.to_sql("logbooks", con=sqlite, if_exists="replace", chunksize=1000)
            first = False
        else:
            chunk.to_sql("logbooks", con=sqlite, if_exists="append", chunksize=1000)
    print("write logbooks done.")
