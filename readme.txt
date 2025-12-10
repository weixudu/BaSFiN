BaSFiN 專案說明

套件部分可直接載入requirements.txt
pip install -r code/requirements.txt


專案結構
本專案主要分為以下幾個資料夾與功能模組：


BaSFiN_code/
├── code/         # 主程式碼與模型
├── data/         # 原始與處理後的特徵資料
├── NAC/          # 論文比較模型之一
├── output/       # 輸出結果
├── plot/         # 圖表繪製
1. code/
開啟 VS Code 時，請將此資料夾設為工作目錄。內含以下子資料夾與檔案：

logs/
儲存實驗結果與訓練紀錄。

model/
模型儲存與載入區，通常不需更動。

BaSFiN/
核心模型與執行代碼，針對 2013–2024 年資料進行實驗：

BaS：主架構。

co_fim：合作特徵交互（觀察兩兩配對分數版本）。

co_fim2：合作特徵交互（一般使用版本）。

bc_fim：競爭特徵交互（觀察兩兩配對分數版本）。

bc_fim2：競爭特徵交互（一般使用版本）。
每個模組均包含對應的 train 代碼與模型定義代碼。

BT/ 與 NAC/
論文中用於比較的模型實作。

BaSFiN_2009_2024/
擴充版，將運行範圍延伸至 2009–2024 年資料。

search/
超參數搜尋（針對 2013–2024 年最佳參數）。

processing/

網頁爬蟲

資料表產生

特徵 tensor 生成

player_id_mapping_2009_2024.csv
玩家 ID 與年份對應表。

2. data/
主要包含：

統計特徵資料的 CSV 表單
用於產生特徵 tensor。

僅包含隊伍成員的 CSV 表單
用於隊局預測。

執行流程

開啟 VS Code 並將 code/ 設為工作資料夾

前置訓練（Pretrain）

先執行 pretrain 相關程式碼，讓模型各自獨立優化。

主要訓練

執行 train_basfin_noInter.py。

重要參數：

force_no_freeze = False：控制是否強制不凍結模型部分權重。

不同年份擴充運行

若需擴充至 2009–2024 年，使用 BaSFiN_2009_2024 內程式碼。

最佳超參數搜尋

使用 search 內程式碼針對 2013–2024 年進行搜尋。

注意事項
model/ 內的檔案通常不需修改，除非需要替換或更新模型權重。

logs/ 資料夾會隨訓練不斷更新，建議定期備份或清理。

不同模組（co_fim, bc_fim 等）對應不同實驗設定，請依研究需求選擇。