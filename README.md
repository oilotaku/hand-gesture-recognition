# hand-gesture-recognition
結合深度資訊的手勢辨識
不同於以往的手勢辨識，通過使用立體相機取得深度資訊，並將深度資訊與色彩圖合併成4維矩陣
再經CNN網路進行分類以達成增加手勢多樣性的效果
實驗中分類了二種手勢三種深度共六類的分組
未使用深度資訊的手勢辨識由於缺少深度資訊無法判斷同一手勢在不同距離的情況
通過將深度資訊導入其中可以判斷同一手勢在不同距離的情況，進而簡化手勢增加分類

立體相機

![圖片1](https://user-images.githubusercontent.com/62131348/160719620-fba4b30e-7881-4679-9d19-3c592463e47f.png)
![圖片2](https://user-images.githubusercontent.com/62131348/160719623-adf4ed9e-a707-4295-b8cd-ec75150d797a.png)
![圖片1](https://user-images.githubusercontent.com/62131348/160718310-fe31c857-7d8d-4d9b-ab6e-2d6449cc7014.png)

CNN訓練比對
![螢幕擷取畫面 2022-03-30 070139](https://user-images.githubusercontent.com/62131348/160720488-06b40796-4e54-4492-ab90-957eecfa85bb.png)
 
 二手勢三深度(六類)
 
 ![圖片1](https://user-images.githubusercontent.com/62131348/160720602-9814006c-5461-47c3-bca5-a3af2dc77300.png)
 
 展示
 
![螢幕擷取畫面 2022-03-30 070816](https://user-images.githubusercontent.com/62131348/160721108-cc4b5c10-0fa2-44d2-a4e6-463c8e57da05.png)
![螢幕擷取畫面 2022-03-30 071227](https://user-images.githubusercontent.com/62131348/160721409-4111f1b5-5939-4774-9742-50f6b50903e3.png)
![螢幕擷取畫面 2022-03-30 071528](https://user-images.githubusercontent.com/62131348/160721674-3efc821f-9b19-423b-944d-e737e8bacad5.png)
![螢幕擷取畫面 2022-03-30 071515](https://user-images.githubusercontent.com/62131348/160721671-c043a4db-1602-4d1f-9cd8-3938600b0eda.png)
