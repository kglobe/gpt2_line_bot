from sqlalchemy import create_engine, text, and_, desc, asc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from models import History, Samples
import time
import pandas as pd
from datetime import datetime

db_url = 'sqlite:///sqlite.db'


def getHistoryById(user_id):
    df = None
    try:
        engine = create_engine(db_url, poolclass = NullPool)
        DB_Session = sessionmaker(bind = engine)
        session = DB_Session()
        resoverall = session.execute(text("SELECT * FROM history WHERE user_id=:user_id ORDER BY datatime "),{"user_id":user_id})
        df = pd.DataFrame(resoverall.fetchall())
        if df.empty == False:
            df.columns = resoverall.keys()
    except Exception as e:
        print('getHistoryById error:',e)
    finally:
        session.close()
        
    return df

def getSamplesById(user_id):
    df = None
    try:
        engine = create_engine(db_url, poolclass = NullPool)
        DB_Session = sessionmaker(bind = engine)
        session = DB_Session()
        resoverall = session.execute(text("SELECT * FROM samples WHERE user_id=:user_id ORDER BY datatime "),{"user_id":user_id})
        df = pd.DataFrame(resoverall.fetchall())
        if df.empty == False:
            df.columns = resoverall.keys()
    except Exception as e:
        print('getSamplesById error:',e)
    finally:
        session.close()
        
    return df

def insertHistory(user_id,ids_array):
    h = History()
    h.user_id = user_id
    h.datatime = time.time()
    h.ids_array = ','.join(str(v) for v in ids_array)
    try:
        engine = create_engine(db_url,poolclass=NullPool)
        DB_Session = sessionmaker(bind=engine)
        session = DB_Session()
        session.add(h)
        session.commit()
    except Exception as e:
        print('insertHistory error:',e)
    finally:
        session.close()

def insertSamples(user_id, speaker, chat_text):
    sp = Samples()
    sp.user_id = user_id
    sp.datatime = time.time()
    sp.speaker = speaker
    sp.chat_text = chat_text
    try:
        engine = create_engine(db_url,poolclass=NullPool)
        DB_Session = sessionmaker(bind=engine)
        session = DB_Session()
        session.add(sp)
        session.commit()
    except Exception as e:
        print('insertSamples error:',e)
    finally:
        session.close()