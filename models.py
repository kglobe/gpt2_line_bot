from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, String, Float, Text, TIMESTAMP
from datetime import datetime
Base = declarative_base()

class History(Base): #向量化歷史對話紀錄
    __tablename__ = 'history'
    user_id = Column(Text, primary_key=True) #LINE ser_id
    datatime = Column(Float, primary_key=True)
    ids_array = Column(Text) #array

class Samples(Base): #歷史對話紀錄
    __tablename__ = 'samples'
    user_id = Column(Text, primary_key=True) #LINE ser_id
    datatime = Column(Float, primary_key=True)
    speaker = Column(Text) #說話的人是誰
    chat_text = Column(Text) #說話內容