# 2020台大土木CAE組與AI中心暑期實習專案 
# AI-Structure-Nonlinear-Analysis
___
## 1.專案介紹
([專案海報連結](https://github.com/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/blob/master/AI%E7%B5%90%E6%A7%8B%E9%9D%9E%E7%B7%9A%E6%80%A7%E6%AD%B7%E6%99%82%E5%88%86%E6%9E%90%E6%9C%9F%E6%9C%AB%E6%B5%B7%E5%A0%B1.pdf))<br>
傳統結構歷時分析採結構力學模型為核心，當結構受震程度較大、進入非線性 (nonlinear)階段後的行為預測計算量甚鉅，
經常耗時數十分鐘甚至數小時不等。使用人工智慧 (Artificial Intelligence, AI) 的機器學習 (machine learning) 提供另一種快速的非線性歷時分析選擇。
本研究以一9層樓鋼構建築為目標結構，訓練並預測結構物在地震作用下的樓層反應歷時。

## 2.結構物非線性反應訓練資料集
`此地震資料集由國家地震中心分析處理與提供`<br>
搜整 1999 年以來七次臺灣地區大型地震紀錄(1999/09/21 九二一集集地震、1999/10/22 嘉義地震、2002/03/31 三三一花蓮外海地震、2006/04/01 四O一台東地震
、2008/12/26 一二二六恆春地震兩次主震、2016/02/06 二O六美濃地震)，以PISA 3D結構工程分析軟體進行近千次結構非線性歷時分析，並經由資料正規化與分類等程序來產生AI訓練所需的資料集。
其中多數地震資料單一地震事件同時擁有經基線修正及未經基線修正兩種版本，目的為增加訓練資料量。<br>
將地震資料之加速度反應譜所對應到目標結構物第一自然振動週期之加速度譜值正規化至0.2g,0.3g,0.4g三個強度等級，初步分類不同非線性程度的地震資料與結構反應資料。

## 3.初期-訓練與結果
以兩層LSTM layer疊加兩層Dense layer作為深度學習模型主要架構，觀察三個不同強度等級的地震資料之預測表現。<br>
`初步觀察結果:`<br>
* 此模型架構下0.2g強度等級的資料預測表現較好
* 結構加速度歷時反應預測結果較結構位移歷時反應好

## 4.後期-針對資料特性進行分析與分類
`此階段針對正規化地震強度0.2g之地震與結構位移反應資料進行近一步研究`<br>
### 1st step
以固有模型架構與資料嘗試調整超參數，與用Ensemble Learning之Bagging方法嘗試提高預測準度，並嘗試在不同運算平台的訓練速度。<br>
&rarr;[`程式碼-in國震中心HPC`](https://github.com/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/blob/master/0.2g_Earthquake/0.2gDisp3/Disp_02g.py)
&[`資料夾`](https://github.com/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/tree/master/0.2g_Earthquake/0.2gDisp3)<br>
&rarr;[`程式碼-in Google Golab`](https://nbviewer.jupyter.org/github/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/blob/master/0.2g_Earthquake/0.2gDisp6/New0.2g_Disp.ipynb)
&[`資料夾`](https://github.com/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/tree/master/0.2g_Earthquake/0.2gDisp6)<br>

### 2nd step
將前步訓練與預測表現好與壞的資料人工分為兩類 &rarr;[`程式碼`](https://nbviewer.jupyter.org/github/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/blob/master/0.2g_Earthquake/DataAnalysis.ipynb)<br>
對兩類地震資料作反應譜分析&rarr;[`程式碼`](https://nbviewer.jupyter.org/github/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/blob/master/0.2g_Earthquake/0.2gDisp9/SpectrumAnalysis.ipynb)<br>
單獨訓練表現好的資料並觀察預測表現&rarr;[`程式碼`](https://github.com/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/blob/master/0.2g_Earthquake/0.2gDisp9/Disp_02g.py)
&[`資料夾`](https://github.com/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/tree/master/0.2g_Earthquake/0.2gDisp9)<br>

### 3rd step
為驗證反應普差異性與資料訓練預測表現相關，使用[`tslearning pakage`](https://tslearn.readthedocs.io/en/stable/index.html)，對基線修正資料的反應譜做
Time series K-Means Clustering，使反應譜資料透過演算法自動分為兩類&rarr;[`程式碼`](https://nbviewer.jupyter.org/github/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/blob/master/Spectrum_Classification/K_means_Clustering.ipynb)<br>
其中一類資料量明顯較多，將此類資料單獨訓練並觀察預測表現&rarr;[`程式碼`](https://nbviewer.jupyter.org/github/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/blob/master/0.2g_Earthquake/0.2gDisp10/Clustering_result.ipynb)
&[`資料夾`](https://github.com/Bosh-Kuo/AI-Structure-Nonlinear-Analysis/tree/master/0.2g_Earthquake/0.2gDisp10)<rb>

## 5.研究結果
從上述試驗發現儘管以地震強度做初步分類，資料間仍存在變異性，導致深度學習模型無法有效預測部分地震資料的結構反應，Time series K-Means Clustering可以以分人為判斷的方式將性質相近的地震資料分群，
且驗證確實同群的資料訓練結果較不分群還要好。<br>
後續由國震中心AI 非線性歷時分析法預測各樓層的準確度均在 95% 以上，預測所需的時間小於 1 秒 — 僅需原本結構動力歷時分析法的 0.2%。針對特定形式或單一重要結構，累積足夠的結構分析資料集後，訓練對應的 AI 模型可預測其在各種地震來襲情境的結構受損機率。預期不久的未來，AI 非線性歷時分析法將扮演風險評估的核心角色。




