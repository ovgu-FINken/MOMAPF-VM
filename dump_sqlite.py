
import sqlalchemy
import pandas as pd

from experiment import get_key

if __name__=="__main__":
    db = sqlalchemy.create_engine(get_key(filename="db.key"))
    sqlite = sqlalchemy.create_engine("sqlite:///sqlite_dump.db")
    
    df = pd.read_sql("jobs", con=db)
    print("read jobs done.")
    df.to_sql("jobs", con=sqlite, if_exists="replace")
    print("write jobs done.")
    df = pd.read_sql("populations", con=db)
    print("read populations done.")
    df.to_sql("populations", con=sqlite, if_exists="replace")
    print("write populations done.")
    df = pd.read_sql("logbooks", con=db)
    print("read logbooks done.")
    df.to_sql("logbooks", con=sqlite, if_exists="replace")
    print("write logbooks done.")
