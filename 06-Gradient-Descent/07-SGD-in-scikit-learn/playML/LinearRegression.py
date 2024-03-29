import numpy as np
from .metrics import r_squared

class LinearRegression:
    def __init__(self):
        self.coef_ = None #系数
        self.intercept_ = None #截距
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to  the size of y_train"
        X_b = np.hstack([np.ones((X_train.shape[0],1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_bgd(self, X_train, y_train, eta = 0.01, n_iters = 1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta))**2) / len(X_b) #X_b是矩阵，theta是向量。矩阵点乘向量结果是列向量。知识点1
            except:
                return float('inf') #如果数字太大返回浮点数最大值
        
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:,i]) #X_b是矩阵，theta是向量。矩阵点成向量结果是列向量，减y以后也是向量。
            #                                                 #向量点乘X_b的列向量，也就是向量点乘向量结果是一个数字。知识点2
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)
        
        def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
            theta = initial_theta
            i_iters = 1
            while i_iters <= n_iters:
                last_theta = theta
                theta = theta - eta * dJ(theta, X_b, y)
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon ):
                    break
                i_iters = i_iters + 1
            return theta
        
        X_b = np.hstack([np.ones((len(X_train),1)),X_train]) #np.ones((len(X),1)).shape是（100，1）,
                                                 #而np.ones((len(X))).shape是（100，）被认为为行向量不能水平合并
        initial_theta = np.zeros(X_b.shape[1])
        eta = 0.01
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta)
        
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型
            n_iters代表循环整个样本几次
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            return 2 * X_b_i.T.dot(X_b_i.dot(theta) - y_i)

        def sgd(X_b, y, initial_theta, n_iters=5, t0=5, t1=50):
            t0, t1 = 5, 50
            def learning_rate(t):
                return t0 / (t + t1)
            
            theta = initial_theta
            m = len(X_b)
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m) #打乱0~m个样本的顺序
                X_b_new = X_b[indexes, :]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient #可以看到越是前面的数据学习率越大，越到后面学习率越小
            return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1]) #randn函数返回一个或一组样本，具有标准正态分布
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self


    def predict(self, X_predict):
        assert self.coef_ is not None and self.intercept_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        
        X_b = np.hstack([np.ones((X_predict.shape[0],1)), X_predict])
        #X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)
    
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r_squared(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"