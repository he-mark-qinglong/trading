
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用agg后端，不展示图形，只保存
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq

class VPVRAnalyzer:
    def __init__(self, df, n_bins=100, n_components=3, volume_col_idx=5,
                 high_col_idx=2, low_col_idx=3, vpvr_dir='vpvr'):
        """
        df: 输入的DataFrame，必须含有价格和成交量列
        n_bins: VPVR价格分区间数
        n_components: GMM的正态分布个数
        volume_col_idx, high_col_idx, low_col_idx: 对应列索引（默认基于用户示范）
        vpvr_dir: 存放VPVR图片文件夹路径
        """
        self.df = df
        self.n_bins = n_bins
        self.n_components = n_components
        self.volume_col_idx = volume_col_idx
        self.high_col_idx = high_col_idx
        self.low_col_idx = low_col_idx
        self.vpvr_dir = vpvr_dir
        
        if not os.path.exists(self.vpvr_dir):
            os.makedirs(self.vpvr_dir)
        
    def compute_vpvr(self):
        low = self.df.iloc[:, self.low_col_idx].values
        high = self.df.iloc[:, self.high_col_idx].values
        volume = self.df.iloc[:, self.volume_col_idx].values
        
        price_min = low.min()
        price_max = high.max()
        bins_edges = np.linspace(price_min, price_max, self.n_bins + 1)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        volume_distribution = np.zeros(self.n_bins)
        
        for h, l, v in zip(high, low, volume):
            if h == l:
                idx = np.searchsorted(bins_edges, h) - 1
                if 0 <= idx < self.n_bins:
                    volume_distribution[idx] += v
            else:
                span = h - l
                overlap_bins = np.where((bins_edges[:-1] < h) & (bins_edges[1:] > l))[0]
                for idx in overlap_bins:
                    bin_low = bins_edges[idx]
                    bin_high = bins_edges[idx + 1]
                    overlap_low = max(bin_low, l)
                    overlap_high = min(bin_high, h)
                    weight = max(0, overlap_high - overlap_low) / span
                    volume_distribution[idx] += v * weight
        
        self.bin_centers = bin_centers
        self.volume_distribution = volume_distribution
    
    def fit_gmm(self):
        vol_dist = self.volume_distribution
        valid_index = vol_dist > 0
        X_centers = self.bin_centers[valid_index]
        V_valid = vol_dist[valid_index]
        
        X = np.repeat(X_centers, V_valid.astype(int)).reshape(-1, 1)
        
        self.gmm = GaussianMixture(n_components=self.n_components,
                                   covariance_type='full',
                                   random_state=42)
        self.gmm.fit(X)
        
        self.means = self.gmm.means_.flatten()
        self.stds = np.sqrt(self.gmm.covariances_).flatten()
        self.weights = self.gmm.weights_
    
    def gaussian(self, x, mean, std, weight=1.0):
        coef = weight / (std * np.sqrt(2 * np.pi))
        return coef * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    def find_intersections(self):
        intersections = []
        for i in range(len(self.means) - 1):
            m1, s1, w1 = self.means[i], self.stds[i], self.weights[i]
            m2, s2, w2 = self.means[i + 1], self.stds[i + 1], self.weights[i + 1]
            def diff(x):
                return self.gaussian(x, m1, s1, w1) - self.gaussian(x, m2, s2, w2)
            
            left = min(m1, m2)
            right = max(m1, m2)
            try:
                cross = brentq(diff, left, right)
                intersections.append(cross)
            except ValueError:
                pass
        
        self.intersections = sorted(intersections)
    
    def mark_regions(self):
        boundaries = [float('-inf')] + self.intersections + [float('inf')]
        regions = []
        for i in range(len(boundaries)-1):
            region = {
                'start': boundaries[i],
                'end': boundaries[i+1],
                'mean': self.means[i] if i < len(self.means) else None,
                'std': self.stds[i] if i < len(self.stds) else None
            }
            regions.append(region)
        
        self.regions = regions
        return regions
    
    def plot_vpvr(self, symbol, filename='vpvr_plot.png', scale_ratio=0.5):  
        vol_dist = self.volume_distribution  
        bin_centers = self.bin_centers  
        means = self.means  
        stds = self.stds  
        weights = self.weights  

        x_min, x_max = bin_centers.min(), bin_centers.max()  
        x_plot = np.linspace(x_min, x_max, 1000)  

        plt.figure(figsize=(12,6))  

        # 画成交量柱  
        plt.bar(bin_centers, vol_dist, width=(bin_centers[1]-bin_centers[0]) * 0.9, alpha=0.4, label='VP Volume')  
        max_bar = vol_dist.max()  

        total_pdf = np.zeros_like(x_plot)  
        for i in range(self.n_components):  
            pdf = weights[i] * (1/(stds[i]*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_plot - means[i])/stds[i])**2)  
            total_pdf += pdf  

        # total_pdf放缩，使其最大高度是max_bar * scale_ratio  
        total_pdf_scaled = total_pdf * max_bar * scale_ratio / total_pdf.max()  
        plt.plot(x_plot, total_pdf_scaled, '--', label='Total Mixture')  

        # 各个高斯分别画，比例同样缩放  
        for i in range(self.n_components):  
            pdf = weights[i] * (1/(stds[i]*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_plot - means[i])/stds[i])**2)  
            pdf_scaled = pdf * max_bar * scale_ratio / total_pdf.max()  
            plt.plot(x_plot, pdf_scaled, label=f'Gaussian {i+1}')  

        for idx, cut in enumerate(self.intersections):  
            plt.axvline(cut, color='red', linestyle='--', label='Cut line' if idx == 0 else "")  

        plt.xlabel('Price')  
        plt.ylabel('Volume')  
        plt.title('VPVR Volume Distribution and GMM Segmentation')  
        plt.legend()  
        plt.tight_layout()  

        if not os.path.exists(self.vpvr_dir):  
            os.makedirs(self.vpvr_dir)  
        timestamp = int(self.df.iloc[-1, 5])  
        filepath = os.path.join(self.vpvr_dir, f'{symbol}_{timestamp}_{filename}')  
        plt.savefig(filepath)  
        plt.close()  
        return filepath  
    
    def run(self, symbol):
        self.compute_vpvr()
        self.fit_gmm()
        self.find_intersections()
        regions = self.mark_regions()
        plot_file = self.plot_vpvr(symbol)
        return {
            'regions': regions,
            'plot_file': plot_file
        }


# 用法示例：
# analyzer = VPVRAnalyzer(coin_date_df, n_bins=100, n_components=3)
# result = analyzer.run()
# print(result['regions'])
# print(f"Plot saved to: {result['plot_file']}")

