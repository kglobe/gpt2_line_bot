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
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import opencc
import get_data_sqlalchemy

PAD = '[PAD]'
pad_id = 0
converter = opencc.OpenCC('s2t.json')

def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成設備')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k選1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高積累概率')
    # parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
    #                     help='模型參數')
    parser.add_argument('--log_path', default='data/interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False, help='選擇詞庫')
    parser.add_argument('--model_path', default='model/epoch40', type=str, required=False, help='對話模型路徑')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天記錄的文件路徑")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重覆懲罰參數，若生成的對話重覆性較高，可適當提高該參數")
    # parser.add_argument('--seed', type=int, default=None, help='設置種子用於生成隨機數，以使得訓練的結果是確定的')
    parser.add_argument('--max_len', type=int, default=25, help='每個utterance的最大長度,超過指定長度則進行截斷')
    parser.add_argument('--max_history_len', type=int, default=3, help="dialogue history的最大長度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU進行預測')
    return parser.parse_args()


def create_logger(args):
    """
    將日志輸出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 創建一個handler，用於寫入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 創建一個handler，用於將日志輸出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
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


def main():
    args = set_args()
    logger = create_logger(args)
    # 當用戶使用GPU,並且GPU可用時
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    # tokenizer = BertTokenizer(vocab_file=args.voca_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天紀錄{}:\n".format(datetime.now()))
    # 存儲聊天記錄，每個utterance以token的id的形式進行存儲
    
    hs_df = get_data_sqlalchemy.getHistoryById('001')
    history = []
    if hs_df is not None and hs_df.empty == False:
        for ids in hs_df.ids_array:
            list_of_strings = ids.split(',')
            history.append(list(map(int, list_of_strings)))
    # print(history)

    while True:
        try:
            text = input("user:")
            # text = "你好"
            get_data_sqlalchemy.insertSamples('001', 'user', text)
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            get_data_sqlalchemy.insertHistory('001',text_ids)
            history.append(text_ids)
            input_ids = [tokenizer.cls_token_id]  # 每個input以[CLS]為開頭

            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            input_ids = torch.tensor(input_ids).long().to(device)
            input_ids = input_ids.unsqueeze(0)
            response = []  # 根據context，生成的response
            # 最多生成max_len個token
            for _ in range(args.max_len):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                # 對於已生成的結果generated中的每個token添加一個重覆懲罰項，降低其生成概率
                for id in set(response):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 對於[UNK]的概率設為無窮小，也就是說模型的預測結果不可能是[UNK]這個token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # torch.multinomial表示從候選集合中無放回地進行抽取num_samples個元素，權重越高，抽到的幾率越高，返回元素的下標
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]則表明response生成結束
                    break
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            # history.append(response)
            
            get_data_sqlalchemy.insertHistory('001',response)
            text = tokenizer.convert_ids_to_tokens(response)
            response_text = "".join(text)
            response_text = converter.convert(response_text)
            print("chatbot:" + response_text)
            get_data_sqlalchemy.insertSamples('001', 'chatbot', response_text)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
