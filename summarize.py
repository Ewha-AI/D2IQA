import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


def show_trend_csv():
    
    csv_list = sorted(glob('../../Desktop/result_new/ge_chest_rep5_c0.08_0.07_0.06/*.csv'))
    # csv_list = sorted(glob('/home/eunbyeol/Desktop/paper_prep/result/dd/*.csv'))
    # csv_list = sorted(glob('../../Desktop/result_new/*mean.csv'))

    count = 0
    for csv_path in csv_list:
        df = pd.read_csv(csv_path, index_col='noise_type')
        df = df['bbox_mAP']

        diff_df = df.diff(periods=-1)[:-1]
        origin_df = pd.read_csv(csv_path, index_col='noise_type')
        
        
        values = diff_df.values.reshape(-1)
        if (np.any(values < 0)):
            count +=1
            print(csv_path)
            print(origin_df)
            csv_filename = csv_path.split('/')[-1]
            # origin_df.to_csv(csv_filename)
            
            # print(origin_df.apply([np.mean, np.std], axis=1))
            # print(values)

    print(count, '/', len(csv_list))


def show_csv():
    
    csv_list = sorted(glob('../../Desktop/result_new/mayo_5/*.csv'))
    # csv_list = sorted(glob('../../Desktop/result_new/*mean.csv'))

    count = 0
    for csv_path in csv_list:
        df = pd.read_csv(csv_path, index_col='Unnamed: 0')[-7:]
        df = df['4']
        diff_df = df.diff(periods=-1)[:-1]
        origin_df = pd.read_csv(csv_path, index_col='Unnamed: 0')[-7:]
        
        
        
        values = diff_df.values.reshape(-1)
        if (np.any(values < 0)):
            count +=1
            print(csv_path)
            print(origin_df)
            csv_filename = csv_path.split('/')[-1]
            # origin_df.to_csv(csv_filename)
            
            # print(origin_df.apply([np.mean, np.std], axis=1))
            # print(values)

    print(count, '/', len(csv_list))


def normalize():
    # csv_list = sorted(glob('../../data/mayo_mmlab_final_41/result/mean/*.csv'))
    csv_list = sorted(glob('../../Desktop/result_new/*mean.csv'))
    count = 0
    for csv_path in csv_list:
        df = pd.read_csv(csv_path, index_col='Unnamed: 0')[:7]
        
        mean_df = df.apply(np.mean, axis=1)
        diff_df = mean_df.diff(periods=-1)[:-1]

        values = diff_df.values.reshape(-1)

        # if (np.any(values < 0)):
        count +=1
        print(csv_path)

        mean_std_df = df.apply([np.mean, np.std], axis=1)
        mus, sigmas = mean_std_df['mean'].values, mean_std_df['std'].values

        legend_list = []
        noise_list = ['full', 'full_0.5', 'quar', 'quar_0.5', 'quar_1.0', 'quar_1.5', 'quar_2.0']
        colors = ['red', 'orange','yellow', 'green', 'cyan', 'blue', 'purple']

        for i in range(len(mus)):
            mu = mus[i]
            sigma = sigmas[i]

            x_axis = np.arange(0, 0.6, 0.001)
            plt.plot(x_axis, norm.pdf(x_axis,mu,sigma), color=colors[i])
            legend_list.append("{}, N({:0.4f}, {:0.4f})".format(noise_list[i], mu, sigma))
            plt.legend(legend_list)  
            plt.xticks(np.arange(0, 0.6, 0.1))

                
        # plt.show()
        plt.savefig(('result/{}.png'.format(count)))
        plt.clf()
        # exit()

def concat_csv():

    csv_list = sorted(glob('/home/eunbyeol/Desktop/result_new/mayo_5_ver2/*_ver2.csv'))
    count = 0

    for csv_path in csv_list:
        df_2 = pd.read_csv(csv_path, index_col='Unnamed: 0')[:7]
        df_2.rename(columns=lambda s: int(s)+5, inplace=True)
                
        df_1 = pd.read_csv((csv_path.replace('/mayo_5_ver2/', '/mayo_5/').replace('_ver2.csv', '.csv')), index_col='Unnamed: 0')[:7]
        df = pd.concat([df_1, df_2], axis=1)
        # df = pd.concat([df_1, df_2], axis=1).iloc[:,1:]

        cum_df = df.cumsum(axis=1)
        mean_df = cum_df.copy()
        for i in range(10):
            mean_df[cum_df.columns[i]] /= (i+1)
        
        # print(mean_df)
        # print(df.diff(axis=1))
        mean_diff = mean_df.diff(axis=1).iloc[:,5:] #300:5, 400:2, 500:1, 600:0
                        
        values = mean_diff.values.reshape(-1)
        if (np.any(abs(values) > 0.01)):
            count +=1
            print(csv_path)
            print(mean_diff)

        print(count, len(csv_list))

if __name__ == '__main__':
    # normalize()
    # show_csv()
    # concat_csv()
    show_trend_csv()
