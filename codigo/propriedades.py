import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dados fornecidos
fluxo_L_min = np.array([20, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0])  # L/min
altura_leito_mm = np.array([55, 55, 55, 53, 50, 50, 50, 50, 50, 50, 50, 50])  # mm
perda_carga_mmca = np.array([5.8, 5.8, 5.8, 5.8, 5.7, 5.6, 5.4, 4.5, 3.5, 3.5, 1.5, 1])  # mmca
estado_leito = ["turbulento", "turbulento", "turbulento", "turbulento",
                "bolhas axiais", "bolhas axiais", "bolhas axiais", "fixo", "fixo", "fixo", "fixo", "fixo"]

# Conversões
Q_m3_s = (fluxo_L_min * 1e-3) / 60  # m³/s
D_m = 0.061  # Diâmetro do leito em metros
A_m2 = np.pi * (D_m ** 2) / 4  # Área da seção transversal

velocidade_m_s = Q_m3_s / A_m2  # Velocidade do ar
perda_carga_Pa = perda_carga_mmca * 9.8  # mmca para Pascal

# Criando o DataFrame
df = pd.DataFrame({
    "Fluxo de Ar (L/min)": fluxo_L_min,
    "Altura do Leito (mm)": altura_leito_mm,
    "Perda de Carga (mmca)": perda_carga_mmca,
    "Perda de Carga (Pa)": perda_carga_Pa,
    "Estado do Leito": estado_leito,
    "Velocidade do ar (m/s)": velocidade_m_s
})

df.round(4)  # Arredondar para melhor apresentação

print (df)