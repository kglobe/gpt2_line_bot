from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
# 載入 LINE Message API 相關函式庫
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
# from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from utility import getConfigBySectionKey

# 載入 json 標準函式庫，處理回傳的資料格式
import json
import opencc
import get_data_sqlalchemy

access_token = getConfigBySectionKey('setting','access_token')
secret = getConfigBySectionKey('setting','secret')
line_bot_api = LineBotApi(access_token)              # 確認 token 是否正確
handler = WebhookHandler(secret)                     # 確認 secret 是否正確

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PAD = '[PAD]'
pad_id = 0

device = '0' #生成設備
temperature = 1.0 #生成的temperature
topk = 8 #最高k選1
topp = 0.0 #最高積累概率
# log_path = 'data/interact.log' #interact日誌存放位置
vocab_path = 'vocab/vocab.txt' #選擇詞庫
model_path = 'model_epoch40_50w' #對話模型路徑
repetition_penalty = 1.0 #重覆懲罰參數，若生成的對話重覆性較高，可適當提高該參數
max_len = 25 #每個utterance的最大長度,超過指定長度則進行截斷
max_history_len = 3 #dialogue history的最大長度
no_cuda = True #不使用GPU進行預測

# def create_logger():
#     """
#     將日志輸出到日志文件和控制台
#     """
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)

#     formatter = logging.Formatter(
#         '%(asctime)s - %(levelname)s - %(message)s')

#     # 創建一個handler，用於寫入日志文件
#     file_handler = logging.FileHandler(
#         filename=log_path)
#     file_handler.setFormatter(formatter)
#     file_handler.setLevel(logging.INFO)
#     logger.addHandler(file_handler)

#     # 創建一個handler，用於將日志輸出到控制台
#     console = logging.StreamHandler()
#     console.setLevel(logging.DEBUG)
#     console.setFormatter(formatter)
#     logger.addHandler(console)

#     return logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最後一維最大的top_k個元素，返回值為二維(values,indices)
        # ...表示其他維度由計算機自行推斷
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 對於topk之外的其他元素的logits值設為負無窮

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 對logits進行遞減排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# logger = create_logger()
# 當用戶使用GPU,並且GPU可用時
cuda = torch.cuda.is_available() and not no_cuda
device = 'cuda' if cuda else 'cpu'
# logger.info('using device:{}'.format(device))
os.environ["CUDA_VISIBLE_DEVICES"] = device
tokenizer = BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
# tokenizer = BertTokenizer(vocab_file=voca_path)
print('load GPT2LMHeadModel...')
model = GPT2LMHeadModel.from_pretrained(model_path)
model = model.to(device)
model.eval()
print('load GPT2LMHeadModel OK!!!')

def chat(user_id, text):
    response_text = None
    hs_df = get_data_sqlalchemy.getHistoryById(user_id)
    history = []
    converter = opencc.OpenCC('s2t.json')
    if hs_df is not None and hs_df.empty == False:
        for ids in hs_df.ids_array:
            list_of_strings = ids.split(',')
            history.append(list(map(int, list_of_strings)))
    try:
        get_data_sqlalchemy.insertSamples(user_id, 'user', text)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        get_data_sqlalchemy.insertHistory(user_id,text_ids)
        history.append(text_ids)
        input_ids = [tokenizer.cls_token_id]  # 每個input以[CLS]為開頭
        for history_id, history_utr in enumerate(history[-max_history_len:]):
            input_ids.extend(history_utr)
            input_ids.append(tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).long().to(device)
        input_ids = input_ids.unsqueeze(0)
        response = []  # 根據context，生成的response
        # 最多生成max_len個token
        for _ in range(max_len):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            # 對於已生成的結果generated中的每個token添加一個重覆懲罰項，降低其生成概率
            for id in set(response):
                next_token_logits[id] /= repetition_penalty
            next_token_logits = next_token_logits / temperature
            # 對於[UNK]的概率設為無窮小，也就是說模型的預測結果不可能是[UNK]這個token
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
            # torch.multinomial表示從候選集合中無放回地進行抽取num_samples個元素，權重越高，抽到的幾率越高，返回元素的下標
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == tokenizer.sep_token_id:  # 遇到[SEP]則表明response生成結束
                break
            response.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
            # print("his_text:{}".format(his_text))
        text = tokenizer.convert_ids_to_tokens(response)
        get_data_sqlalchemy.insertHistory(user_id,response)
        response_text = "".join(text)
        response_text = converter.convert(response_text)
        # print("chatbot:" + response_text)
        get_data_sqlalchemy.insertSamples(user_id, 'chatbot', response_text)
    except Exception as e:
        print('linebot chat:',e)
    
    return response_text

@app.get("/check")
def read_root():
    return 'OK'

@app.post("/repeat")
async def linebot(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()                    # 取得收到的訊息內容
    try:
        json_data = json.loads(body)                         # json 格式化訊息內容
        handler.handle(body.decode("utf-8"), x_line_signature)                      # 綁定訊息回傳的相關資訊
        msg = json_data['events'][0]['message']['text']      # 取得 LINE 收到的文字訊息
        tk = json_data['events'][0]['replyToken']            # 取得回傳訊息的 Token
        user_id = json_data['events'][0]['source']['userId'] #取得使用者id
        # print(user_id,':' ,msg)                                # 印出內容
        line_bot_api.reply_message(tk,TextSendMessage(msg))  # 回傳訊息
    except:
        print(body)                                          # 如果發生錯誤，印出收到的內容
    return 'OK'                 # 驗證 Webhook 使用，不能省略

@app.post("/")
async def linebot(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()                    # 取得收到的訊息內容
    try:
        json_data = json.loads(body)                         # json 格式化訊息內容
        handler.handle(body.decode("utf-8"), x_line_signature)                      # 綁定訊息回傳的相關資訊
        msg = json_data['events'][0]['message']['text']      # 取得 LINE 收到的文字訊息
        tk = json_data['events'][0]['replyToken']            # 取得回傳訊息的 Token
        user_id = json_data['events'][0]['source']['userId'] #取得使用者id
        # print(user_id,':' ,msg)                              #印出使用者id跟內容
        response_text = chat(user_id,msg)
        line_bot_api.reply_message(tk,TextSendMessage(response_text))  # 回傳訊息
    except Exception as e:
        print('linebot except:',e)                                      # 如果發生錯誤，印出收到的內容
    return 'OK'                 # 驗證 Webhook 使用，不能省略

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))