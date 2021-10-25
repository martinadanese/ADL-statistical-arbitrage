# Statistical Arbitrage

This is a repository for the project of Applied Deep Learning course @ TUWien. As suggested by the name, the topic of this project is _Statistical Arbitrage_, and the type is _Beat the stars_.

## Idea and Approach

### Introduction
Statistical Arbitrage (SA) is a short-term algorithmic trading approach that aims at extracting a time-series signals of a portfolio of similar assets, in order to capture deviations. When the deviation is strong enough, it is possible to apply a profitable trading policy, going long (buying) one or more stocks and going short (selling) an equal amount according to the sign of the return spread between single stocks and their median. It is worth noticing that SA does not depend on absolute stock prices, being it is market-neutral, but the result is deeply affected by a precise modelling of stocks relationships and open and close position.

Research in finance and trading has a dichotomic nature, being split into pure financial research and computer science, the latter being more and more dominated by deep learning (DL) approach [1-3]. Such trend is observed also in SA subfield, where recurrent neural networks (RNN) [1,2,4] and reinforcement learning (RL) [1-3] have been dominant. However, recently the focus was set on the time-series nature of the data, leading to the first adoption of long short-term memory (LSTM) networks [5,6], which was proven to outperform ARIMA models [7] and random forests when adopting a multi-feature setting consisting of returns with respect to the closing, opening prices and intraday returns [8].

On the other hand, newly developed techniques of time-dependent DL such as different attention mechanisms have not been adopted for SA up to my knowledge. A suitable attention mechanism has been proven to impact results in many fields, such as natural language [9] and image processing [10]. 
Thus, in my project I aim at including a suitable attention mechanism to the workflow build in refs. [5-8]. Specifically I will attempt to introduce and compare soft-attention mechanism [10], and the hybrid hard and soft attention for sequence modeling introduced by Shen et al. [12].


### Methods

The adopted procedure follows the steps sketched by Krauss et al. [5], which were recently proven profitable for machine learning-based predictions for SA, and in particular for LSTM approach [6,8]. To compare results, the same dataset will be used, namely all stocks of the S&P 500 from the period of January 1990 until December 2018. However, it might turn necessary to decrease the periods or number of stocks because of the limited disposable computational power.

*  __Preprocessing__. In this phase, the dataset is divided in study periods, each in turn sub-divided into a feature creation, a training and a testing (or trading) part, the latter being non overlapping for each study period. For each stock _s_ and each study period a vector of feature is created computing intraday return (_ir_), returns with respect to last closing price (_cr_) and returns with respect to opening price (_op_), defined as ratios between the current timestamp _t_ and each _m_ timestep of the feature part. These are standardized and merged into a feature matrix (Figure from [8]):

<p align="center">
<img src="https://user-images.githubusercontent.com/86531192/138686100-19c78a9d-5e25-4c11-910a-51710900cc57.png" width="500">
</p>

* __Training__. In this phase I will train two LSTM networks, starting with a structure analogous to the ones described in refs. [5-8], one network implementing soft attention mechanism and one implementing hybrid attention mechanism. The model parameters will be changed, thus the result might be considerably different compared to the one of the cited papers.

* __Trading__. The trading policy will be to go long the 10 stocks with highest probability to outperform the median intraday return and go short the 10 stocks with the lowest probability of such an outcome. If time allows it, new policies might be attempted, such as neural networks mapping of the arbitrage signals [13].



## work-breakdown structure
* __dataset collection__. _Method_: quandl library/github repositories. _Time_: 2h;
* __preprocessing__. _Method_: python. _Time_: 8h-10h;
* __designing and building an appropriate network__. _Method_: python (mainly keras). _Time_: 8h-10h;
* __training and fine-tuning that network__. _Method_: python (mainly keras). _Time_: 8h-10h;
* __building an application to present the results__. _Method_: python. _Time_: 8h-10h;
* __writing the final report__. _Method_: LaTeX. _Time_: 8h;
* __preparing the presentation__. _Method_: tbd. _Time_: 8h;

__Total time__: 50-58h


## Bibliography

[1] Zhang, Z., Zohren, S., & Roberts, S. (2020). _Deep reinforcement learning for trading_. The Journal of Financial Data Science, 2(2), 25-40.

[2] Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2016). _Deep direct reinforcement learning for financial signal representation and trading_. IEEE transactions on neural networks and learning systems, 28(3), 653-664

[3] Sun, S., Wang, R., & An, B. (2021). _Reinforcement Learning for Quantitative Trading_. arXiv:2104.14214.

[4] Huang, J., Chai, J., & Cho, S. (2020). _Deep learning in finance and banking: A literature review and classification_. Frontiers of Business Research in China, 14, 1-24.



[5] Krauss, C., Do, X. A., & Huck, N. (2017). _Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500_. European Journal of Operational Research, 259, 689–702.

[6] Fischer, T., & Krauss, C. (2018). _Deep learning with long short-term memory networks for financial market predictions._ European Journal of Operational Research, 270, 654–669.

[7] Siami-Namini, S., & Namin, A. S. (2018). Forecasting economics and financial time series: ARIMA vs. LSTM. preprint ,
arXiv:1803.06386.

[8] Ghosh, P., Neufeld, A., & Sahoo, J. K. (2021). _Forecasting directional movements of stock prices for intraday trading using LSTM and random forests_. Finance Research Letters, 102280.






[9] Jain, D., Kumar, A., & Garg, G. (2020). _Sarcasm detection in mash-up language using soft-attention based bi-directional LSTM and feature-rich CNN_. Applied Soft Computing, 91, 106198.

[10] Chu, Y., Yue, X., Yu, L., Sergei, M., & Wang, Z. (2020). _Automatic image captioning based on ResNet50 and LSTM with soft attention_. Wireless Communications and Mobile Computing, 2020.







[11] Bahdanau, D., Cho, K., & Bengio, Y. (2014). _Neural machine translation by jointly learning to align and translate_. arXiv:1409.0473.

[12] Shen, T., Zhou, T., Long, G., Jiang, J., Wang, S., & Zhang, C. (2018). _Reinforced self-attention network: a hybrid of hard and soft attention for sequence modeling_. arXiv preprint arXiv:1801.10296.

[13] Guijarro-Ordonez, J., Pelger, M., & Zanotti, G. (2021). _Deep Learning Statistical Arbitrage_. arXiv:2106.04028 





