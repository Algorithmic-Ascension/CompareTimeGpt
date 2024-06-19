# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:35:52 2024

@author: camer
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np

GDP = pd.read_csv("./gdp.csv", sep=",", header=0, usecols=list(range(1, 34)))
Cons = pd.read_csv("./consumption.csv", sep=",", header=0, usecols=list(range(1, 34)))
Un = pd.read_csv("./unemployment.csv", sep=",", header=0, usecols=list(range(1, 34)))

mse1 = [[] for each in range(6)]

for i in range(6):

    vec = np.zeros((66, 1))
    curlen = 0

    for j in range(33):
        Train_x = GDP.iloc[: -3 - i, j]
        Train_y = GDP.iloc[1 + i : -2, j]
        Test_x = sm.add_constant(GDP.iloc[-3 - i : -1 - i, j])
        Test_y = GDP.iloc[-2:, j]
        idx = Test_y[Test_y < -99].index
        Test_y = Test_y.drop(idx)
        Test_x = Test_x.drop(idx - 1 - i)

        Train_x = sm.add_constant(Train_x).to_numpy()
        Test_x = Test_x.to_numpy()
        model = sm.OLS(Train_y, Train_x)
        results = model.fit()
        pars = results.params.to_numpy()
        Test_y = Test_y.to_numpy()
        out = np.matmul(pars.reshape((1, 2)), Test_x.T)
        MSE = (Test_y - out) ** 2
        MSE = MSE[0]
        length = len(MSE)
        vec[curlen : curlen + length, 0] = MSE
        curlen = curlen + length
    mse1[i] = vec[:curlen, 0]
    print(np.sqrt(np.average(mse1[i])))

"""
RMSE:Invidivdual AR(1)
H1: 0.042799612800105954
H2: 0.043419836711727666
H3: 0.0419687947164618
H4: 0.043702659214746385
H5: 0.04391401990822678
H6: 0.04333081023093387
"""

for i in range(6):

    vec = np.zeros((66, 1))
    curlen = 0

    for j in range(33):
        if j == 0:
            Train_x = (sm.add_constant(GDP.iloc[: -3 - i, j])).to_numpy()
            Train_y = GDP.iloc[1 + i : -2, j].to_numpy()
            Test_x = sm.add_constant(GDP.iloc[-3 - i : -1 - i, j])
            Test_y = GDP.iloc[-2:, j]
            idx = Test_y[Test_y < -99].index
            Test_y = Test_y.drop(idx)
            Test_x = Test_x.drop(idx - 1 - i)
            Test_x = Test_x.to_numpy()
            Test_y = Test_y.to_numpy()

        else:
            Train_x_t = (sm.add_constant(GDP.iloc[: -3 - i, j])).to_numpy()
            Train_y_t = GDP.iloc[1 + i : -2, j].to_numpy()
            Test_x_t = sm.add_constant(GDP.iloc[-3 - i : -1 - i, j])
            Test_y_t = GDP.iloc[-2:, j]
            idx = Test_y_t[Test_y_t < -99].index
            Test_y_t = Test_y_t.drop(idx)
            Test_x_t = Test_x_t.drop(idx - 1 - i)
            Test_x_t = Test_x_t.to_numpy()
            Test_y_t = Test_y_t.to_numpy()
            Train_x = np.concatenate([Train_x, Train_x_t], axis=0)
            Train_y = np.concatenate([Train_y, Train_y_t], axis=0)
            Test_x = np.concatenate([Test_x, Test_x_t], axis=0)
            Test_y = np.concatenate([Test_y, Test_y_t], axis=0)

    # Train_x = sm.add_constant(Train_x).to_numpy()
    # Test_x = Test_x.to_numpy()
    # model = sm.OLS(Train_y,Train_x)
    # results = model.fit()
    # pars = results.params.to_numpy()
    # Test_y = Test_y.to_numpy()
    # out = np.matmul(pars.reshape((1,2)),Test_x.T)
    # MSE = (Test_y-out)**2
    model = sm.OLS(Train_y, Train_x)
    results = model.fit()
    pars = results.params
    out = np.matmul(pars.reshape((1, 2)), Test_x.T)
    MSE = (Test_y - out) ** 2
    MSE = MSE[0]

    mse1[i] = MSE
    print(np.sqrt(np.average(mse1[i])))
"""
RMSE: Pooled AR(1)
H1: 0.04235479271965332
H2: 0.04287406556455985
H3: 0.042073386377908445
H4: 0.04279915604687209
H5: 0.04279861459519815
H6: 0.04283785888115586

"""

# VAR(1) Individual
for i in range(6):

    vec = np.zeros((66, 1))
    curlen = 0

    for j in range(33):
        Train_x = pd.concat(
            [GDP.iloc[: -3 - i, j], Cons.iloc[: -3 - i, j], Un.iloc[: -3 - i, j]],
            axis=1,
            ignore_index=True,
        )
        Train_y = GDP.iloc[1 + i : -2, j]
        Test_x = sm.add_constant(
            pd.concat(
                [
                    GDP.iloc[-3 - i : -1 - i, j],
                    Cons.iloc[-3 - i : -1 - i, j],
                    Un.iloc[-3 - i : -1 - i, j],
                ],
                axis=1,
                ignore_index=True,
            ),
            has_constant="add",
        )
        Test_y = GDP.iloc[-2:, j]
        idx = Test_y[Test_y < -99].index
        Test_y = Test_y.drop(idx)
        Test_x = Test_x.drop(idx - 1 - i)

        Train_x = sm.add_constant(Train_x).to_numpy()
        Test_x = Test_x.to_numpy()
        model = sm.OLS(Train_y, Train_x)
        results = model.fit()
        pars = results.params.to_numpy()
        Test_y = Test_y.to_numpy()
        out = np.matmul(pars.reshape((1, 4)), Test_x.T)
        MSE = (Test_y - out) ** 2
        MSE = MSE[0]
        length = len(MSE)
        vec[curlen : curlen + length, 0] = MSE
        curlen = curlen + length
    mse1[i] = vec[:curlen, 0]
    print(np.sqrt(np.average(mse1[i])))
"""
RMSE
H1: 0.04384357870287875
H2: 0.043780583054472334
H3: 0.04204334826825148
H4: 0.04651158613308181
H5: 0.044494352814250085
H6: 0.043157141319761644
"""
# VAR(1) Pooled
for i in range(6):

    vec = np.zeros((66, 1))
    curlen = 0

    for j in range(33):
        if j == 0:
            Train_x = sm.add_constant(
                pd.concat(
                    [
                        GDP.iloc[: -3 - i, j],
                        Cons.iloc[: -3 - i, j],
                        Un.iloc[: -3 - i, j],
                    ],
                    axis=1,
                    ignore_index=True,
                )
            )
            Train_y = GDP.iloc[1 + i : -2, j]
            Test_x = sm.add_constant(
                pd.concat(
                    [
                        GDP.iloc[-3 - i : -1 - i, j],
                        Cons.iloc[-3 - i : -1 - i, j],
                        Un.iloc[-3 - i : -1 - i, j],
                    ],
                    axis=1,
                    ignore_index=True,
                ),
                has_constant="add",
            )
            Test_y = GDP.iloc[-2:, j]
            idx = Test_y[Test_y < -99].index
            Test_y = Test_y.drop(idx)
            Test_x = Test_x.drop(idx - 1 - i)
            Test_x = Test_x.to_numpy()
            Test_y = Test_y.to_numpy()

        else:
            Train_x_t = sm.add_constant(
                pd.concat(
                    [
                        GDP.iloc[: -3 - i, j],
                        Cons.iloc[: -3 - i, j],
                        Un.iloc[: -3 - i, j],
                    ],
                    axis=1,
                    ignore_index=True,
                )
            )
            Train_y_t = GDP.iloc[1 + i : -2, j]
            Test_x_t = sm.add_constant(
                pd.concat(
                    [
                        GDP.iloc[-3 - i : -1 - i, j],
                        Cons.iloc[-3 - i : -1 - i, j],
                        Un.iloc[-3 - i : -1 - i, j],
                    ],
                    axis=1,
                    ignore_index=True,
                ),
                has_constant="add",
            )
            Test_y_t = GDP.iloc[-2:, j]
            idx = Test_y_t[Test_y_t < -99].index
            Test_y_t = Test_y_t.drop(idx)
            Test_x_t = Test_x_t.drop(idx - 1 - i)
            Test_x_t = Test_x_t.to_numpy()
            Test_y_t = Test_y_t.to_numpy()
            Train_x = np.concatenate([Train_x, Train_x_t], axis=0)
            Train_y = np.concatenate([Train_y, Train_y_t], axis=0)
            Test_x = np.concatenate([Test_x, Test_x_t], axis=0)
            Test_y = np.concatenate([Test_y, Test_y_t], axis=0)

    # Train_x = sm.add_constant(Train_x).to_numpy()
    # Test_x = Test_x.to_numpy()
    # model = sm.OLS(Train_y,Train_x)
    # results = model.fit()
    # pars = results.params.to_numpy()
    # Test_y = Test_y.to_numpy()
    # out = np.matmul(pars.reshape((1,2)),Test_x.T)
    # MSE = (Test_y-out)**2
    model = sm.OLS(Train_y, Train_x)
    results = model.fit()
    pars = results.params
    out = np.matmul(pars.reshape((1, 4)), Test_x.T)
    MSE = (Test_y - out) ** 2
    MSE = MSE[0]

    mse1[i] = MSE
    print(np.sqrt(np.average(mse1[i])))
"""
RMSE:
H1: 0.04202691971362027
H2: 0.0426593039465102
H3: 0.04164656561510848
H4: 0.042507587598604894
H5: 0.04231372775144572
H6: 0.042252436135685655
"""
