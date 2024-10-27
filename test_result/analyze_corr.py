import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import scipy.stats
import os

def noise2score():

    ans_path = '../../data/doc_test/ans.csv'
    ans_df = pd.read_csv(ans_path, index_col='title')
    ans_idx = ans_df.index
    noise_dict = {'full': 0,
                'full_0.5': 1,
                'quarter' : 2,
                'quarter_0.5' : 3,
                'quarter_1.0' : 4,
                'quarter_1.5' : 5,
                'quarter_2.0' : 6}  

    rows = []
    for i in range(len(ans_df)):
        row = list(ans_df.iloc[i, :])
        idx = ans_idx[i]


        csv_path = '../../Desktop/result_new/mayo_5_quarter/result_quarter_mAP_mayo_5_{}_mean.csv'.format(idx)
        df = pd.read_csv(csv_path, index_col='Unnamed: 0')[-7:]

        temp_values = list(np.round(df['4'].values,4))
        # values = list(map(lambda x: x*20, temp_values))
        values = list(map(lambda x: x*1, temp_values))

        new_row = [idx] + [values[noise_dict[i]] for i in row]
                    
        rows.append(new_row)

    csv = pd.DataFrame(rows, columns=['name', '1', '2', '3', '4', '5'])
    csv.to_csv('test_result/quarter2score.csv', index=False)
        

def visualize_comparison(target_path, label_path):

    pred_df = pd.read_csv(target_path, index_col=0)
    doct_df = pd.read_excel(label_path, index_col='score')    

    pred_list = []
    doct_list = []
    
    pearsonr = 0
    spearmanr = 0

    img_list = doct_df.index
    fig = plt.figure(figsize=(15,10))

    for i in range(30):
        img = img_list[i]
        pred = pred_df.loc[img].values
        doct = doct_df.loc[img].values[:5]

        pred_list.append(pred)
        doct_list.append(doct)

        pearsonr += scipy.stats.pearsonr(pred, doct)[0]
        spearmanr += scipy.stats.spearmanr(pred, doct).correlation

        #Visualize
    #     ax = fig.add_subplot(5,6,i+1)
    #     plt.plot([1,2,3,4,5], pred, 'r')
    #     plt.plot([1,2,3,4,5], doct, 'g')
    # plt.show()
    return pred_list, doct_list, pearsonr/30, spearmanr/30

def mo2score():
    
    csv_path = 'test_result/result_mo.csv'
    csv_df = pd.read_csv(csv_path, index_col='dataset')

    mo_df = csv_df[csv_df['value'] == 'SNR5']
    mo_df.rename(columns = {"dataset": "name"}, inplace=True)
    mo_df.drop(columns='value', inplace=True)
    mo_df.to_csv('test_result/mo2score.csv')


def calculate_corr(target_path, label_path):

    pred_list, doct_list, pearsonr, spearmanr = visualize_comparison(target_path, label_path)

    doct_np = np.array(doct_list, dtype=np.float32).reshape(-1)
    pred_np = np.array(pred_list, dtype=np.float32).reshape(-1)
    
    print('< {} >'.format(target_path.split('2')[0]))
    print('평균 피어슨 상관계수 : %5.4f' %pearsonr)
    print('평균 스피어만 상관계수 : %5.4f' %spearmanr)
    print('-------------------------------')
    print('종합 피어슨 상관계수 : %5.4f' %scipy.stats.pearsonr(pred_np, doct_np)[0])
    print('종합 스피어만 상관계수 : %5.4f\n' %scipy.stats.spearmanr(pred_np, doct_np).correlation)

    # plt.plot(doct_np,pred_np, 'o')
    # plt.show()


if __name__ == '__main__':



    doct_path1 = 'test_result/result_mela.xlsx'
    doct_path2 = 'test_result/result_beck.xlsx'

    #Correlation between doctor and model observer(MO).
    # target_path = 'test_result/mo2score.csv'
    # if not os.path.isfile(target_path):
    #     mo2score()
    # calculate_corr(target_path, doct_path1)
    # calculate_corr(target_path, doct_path2)
    

    #Correlation between doctor and our metric.
    target_path = 'test_result/fullquarter2score.csv'    
    if not os.path.isfile(target_path):
        noise2score()
    calculate_corr(target_path, doct_path1)
    calculate_corr(target_path, doct_path2)
    exit()

    
    #Correlation between doctor and NRQM metrics.
    # nr_target_path = ['test_result/nr_result_brisque.csv',
    #                 'test_result/nr_result_niqe.csv',
    #                 'test_result/nr_result_piqe.csv']
    # for target_path in nr_target_path:
    #     calculate_corr(target_path)

    




    
