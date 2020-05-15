#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Este código contém o ajuste de uma função logística (solução do modelo SI)
    aos dados oficiais do total de pessoas infectadas por COVID-19 no Brasil até
    14/05/20.
    Autor: Guilherme Ricioli Cruz <guilherme.riciolic@gmail.com>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Lendo os dados oficiais de pessoas infectadas pela COVID-19
# (fonte: Ministério da Saúde <https://covid.saude.gov.br/>)
# ##############################################################################
data = np.array(pd.read_csv(r'data/infected-05-14.csv', header=None))
data = np.ndarray.flatten(data)
days = np.arange(len(data))

# Definindo a função logística (solução do modelo SI - suceptíveis - infectados)
# ##############################################################################
def logistic_func(t, alpha, gamma):
    M = 211515650   # População brasileira em 14/05/2020 @ 23:06:49 (fonte: IBGE
                    # <https://www.ibge.gov.br/apps/populacao/projecao/box_popclock.php>)
    return 1/((1/(alpha*M)) + ((1/data[0]) - (1/(alpha*M)))*np.exp(-gamma*alpha*t))

# Ajustando os parâmetros 'alpha' e 'gamma' da função logística aos dados reais
# ##############################################################################
popt, pcov = curve_fit(logistic_func, days, data, bounds=(0,[1.,300.0]))

# Plotando os dados oficiais e a curva logística ajustada ao mesmos
# ##############################################################################
plt.scatter(days, data, marker='.', color='red', label ='Dados oficiais')
plt.plot(logistic_func(days, *popt), label='Curva logística ajustada')
plt.text(30, 150000, 'alpha=%1.4f, gamma=%5.1f' % tuple(popt))
plt.title('Dados oficiais e curva logística ajustada aos mesmos')
plt.xlabel('Tempo transcorrido desde a primeira infecção [em dias]')
plt.ylabel('Total de pessoas infectadas')
plt.legend()
plt.show()
