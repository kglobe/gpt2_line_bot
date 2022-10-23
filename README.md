# GPT2 LINE BOT 閒聊機器人

## 使用模型
使用【yangjianxin1/GPT2-chitchat】(https://github.com/yangjianxin1/GPT2-chitchat)此項目，將訓練好的模型，綁到LINE BOT上面

## LINE BOT
需要將LINE BOT的access_token跟secret，設定到config.ini才能用自己的機器人發訊息
且webhook需要將app engine的網址寫上

## 運行環境
可以參考【yangjianxin1/GPT2-chitchat】此專案的運行環境
python3.6、 transformers==4.2.0、pytorch==1.7.0、line-bot-sdk==2.3.0、fastapi==0.70.1

## 模型下載
直接到【yangjianxin1/GPT2-chitchat】(https://github.com/yangjianxin1/GPT2-chitchat)介紹中有連結可以下載
【model_epoch40_50w】此模型資料夾

## 網頁伺服器(Webhook)
使用fastapi架在Google App Engine上來跑
而且模型是在server啟動時即載入，避免每次request都重新載入，時間過久
使用sqlite來記錄不同的id的對話，讓對話預測有過往聊天內容

## TODO
- 台灣用語過少，都是文章的語料過多
- 使用interact.py來重新訓練對話紀錄

## Reference
- [LINE BOT設定](https://github.com/FawenYo/LINE_Bot_Tutorial)
- [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
- [Deploy FastAPI on Google Cloud Platform](https://github.com/windson/fastapi/tree/fastapi-deploy-google-cloud-platform)

- [DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)

## 一些GCP指令筆記
- [設定專案] gcloud config set project {project name}

- [Build Image] gcloud builds submit --tag gcr.io/{project name}/{container name}

- [部屬image] gcloud run deploy --image gcr.io/talking-fish-webhook/quickstart-image:v14 --platform managed

- [部屬APP Engine] gcloud app deploy app.yaml
- 在線上Console使用部屬APP就可以，檔案直接網頁上傳即可