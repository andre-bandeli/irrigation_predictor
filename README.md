# Modelagem Preditiva Para Determinação do Consumo de Água em Lavouras Utilizando Redes Neurais Artificiais e Random Forest Regressor

**André Luiz Bandeli Júnior; Luiz Fernando Ferreira da Silva Neto; Ronaldo Lopes da Silva Filho**

Faculdade de Engenharia Agrícola da Unicamp
Introdução à Mineração de Dados Aplicados à Agricultura

---
### Resumo
A utilização de Deep Learning em problemas relacionados à agricultura tem crescido nos últimos anos, tornando-se uma ferramenta relevante de predição. Este trabalho apresenta a implementação de uma **Rede Neural Artificial (RNA)** e um modelo de **Random Forest Regressor (RFR)** para prever o consumo de água com base em sete variáveis: tipo de cultura, área da fazenda, tipo de irrigação, uso de fertilizantes, uso de pesticidas, produção total e tipo de solo.

A metodologia utilizada incluiu um processo **KDD (Descoberta de Conhecimento em Dados)**, com análise exploratória e remoção de nulos e *outliers*. O *dataset* foi dividido em 80% para treino e 20% para teste. O modelo Random Forest apresentou melhor desempenho com $R^{2}$ de 0,98 em uma base de dados aumentada, superando o resultado da Rede Neural Artificial, que obteve $R^{2}$ de 0,80.

---
### Introdução
A análise de dados sobre o consumo de água é crucial para a agricultura, especialmente para o manejo e otimização da irrigação. Devido à alta demanda de água na agricultura e à preocupação global com os recursos hídricos, é essencial adotar estratégias que permitam a economia de água sem afetar a produtividade. O uso de abordagens de Machine Learning e Deep Learning tem se intensificado para identificar padrões e fazer previsões mais precisas que auxiliem na tomada de decisão no setor agrícola.

O objetivo deste trabalho é apresentar uma análise descritiva de dados de irrigação e um modelo preditivo para o consumo de água, utilizando Redes Neurais Artificiais.

---
### Objetivos
O processo do projeto seguiu as seguintes etapas:
* Download e conversão da base de dados para o formato .xlsx.
* Aplicação da metodologia de Descoberta de Conhecimento em Dados (KDD) para preparação e análise dos dados.
* Implementação e avaliação de um modelo de Rede Neural Artificial.
* Desenvolvimento de um *script* para realizar predições.
* Apresentação e discussão dos resultados, com foco no desempenho e aplicabilidade do modelo.

---
### Metodologia
A metodologia do estudo envolveu o download, pré-processamento, aplicação de KDD, e implementação da rede neural e do preditor.

#### Base de Dados
A base de dados sobre irrigação e consumo de água foi selecionada por sua relevância na análise da eficiência hídrica em sistemas agrícolas. As análises preditivas relacionaram variáveis como tipo de irrigação, área cultivada, uso de insumos, produtividade e características do solo com o volume de água utilizado. As variáveis selecionadas para o estudo são detalhadas na Tabela 1.

| Farm_ID | CT | FA (acres) | IT | FU (ton) | PU (tons) | Yield (tons) | ST | WU (m³) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| F001 | Cotton | 329.4 | Sprinkler | 8.14 | 2.21 | 14.44 | Loamy | 76648.2 |
| F002 | Carrot | 18.67 | Manual | 4.77 | 4.36 | 42.91 | Peaty | 68725.54 |
| F003 | Sugarcane | 306.03 | Flood | 2.91 | 0.56 | 33.44 | Silty | 75538.56 |
_Legenda: CT = Crop Type, FA = Farm Area, IT = Irrigation System, FU = Fertilizer Used, PU = Pesticide Used, ST = Soil Type e WU = Water Used._

#### Processamento dos Dados
O pré-processamento dos dados incluiu: tratamento de valores faltantes e nulos, remoção de *outliers* com base na análise dos quartis e transformação dos dados. Foi aplicado um processo KDD para extrair mais informações da base de dados.

#### Redes Neurais Artificiais (RNA)
As RNAs são sistemas que mapeiam vetores de entrada em vetores de saída, inspirados em sistemas biológicos humanos. A modelagem foi realizada com a biblioteca `scikit-learn` em Python, utilizando o **MLPRegressor**. A otimização de hiperparâmetros foi feita com **GridSearch**. A Tabela 2 lista os hiperparâmetros testados.

| Hiperparâmetro | Valores testados |
| :--- | :--- |
| `hidden_layer_sizes` | `(50,)`, `(100,)`, `(100, 50)`, `(150, 100, 50)` |
| `alpha` | `0.0001`, `0.001`, `0.01` |
| `activation` | `ReLU`, `Tanh` |
| `learning_rate` | `Constant`, `Adaptive` |

#### Random Forest
O algoritmo Random Forest é uma técnica de aprendizado de máquina que combina árvores de decisão independentes para aumentar a capacidade de generalização do modelo. O modelo foi implementado com a biblioteca `scikit-learn` do Python. A otimização de hiperparâmetros também foi realizada com a técnica GridSearch. A Tabela 3 mostra os parâmetros utilizados.

| Hiperparâmetro | Valores testados |
| :--- | :--- |
| `n_estimators` | `100`, `200`, `300` |
| `max_depth` | `10`, `20`, `None` |
| `min_samples_split` | `2`, `5`, `10` |
| `min_samples_leaf` | `1`, `2`, `4` |

#### Geração de Dados Sintéticos
Para superar as limitações da base de dados original, uma expansão foi realizada utilizando a técnica de *bootstrap*. Esta abordagem criou um volume adicional de registros que replicam as distribuições estatísticas dos dados iniciais.

---
### Resultados e Análise
Inicialmente, os modelos apresentaram desempenho baixo devido à pequena quantidade de dados. A Tabela 4 mostra os resultados iniciais dos modelos.

| Modelo | MSE | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | 9.517.950.059.921 | 308.511.751 | 250.370.269 | -1.963 |
| Rede Neural | 9.335.039.905.310 | 305.532.975 | 240.249.707 | -1.733 |

Após a expansão da base com dados sintéticos, a performance dos modelos melhorou substancialmente. O Random Forest alcançou um $R^{2}$ de 0,98, enquanto a Rede Neural obteve 0,80.

| Modelo | MSE | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | 12172103.6864 | 3488.8542 | 2343.1955 | 0.9817 |
| Rede Neural | 126580423.2869 | 11250.7966 | 8774.3947 | 0.8097 |

---
### Conclusões
1.  A base de dados sobre consumo de água na agricultura se mostrou relevante para estudos técnicos que visam a conservação da água.
2.  O pré-processamento, a análise exploratória de dados e a aplicação de KDD foram essenciais para identificar padrões e inconsistências. A geração de dados sintéticos foi fundamental para mitigar a limitação da amostra original.
3.  A Rede Neural Artificial demonstrou potencial para capturar relações não lineares, mas foi sensível à escassez de dados, resultando em baixa acurácia na base original.
4.  O modelo Random Forest apresentou melhor desempenho em todos os indicadores avaliados, mostrando maior robustez e menor propensão ao *overfitting* em cenários com dados reduzidos.
5.  A estratégia de ampliação da base com dados sintéticos impactou positivamente o desempenho, especialmente do Random Forest, que alcançou valores elevados de $R^{2}$.
6.  O modelo de árvores de decisão (Random Forest) foi superior neste estudo, especialmente em termos de estabilidade e generalização.
7.  O uso de modelos de Machine Learning pode contribuir significativamente para a previsão do consumo hídrico na agricultura, desde que haja bases de dados consistentes.

---
### Referências Bibliográficas
* **BEJANI**, Mohammad Mahdi; **GHATEE**, Mehdi. Regularized deep networks in intelligent transportation systems: a taxonomy and a case study. **Artificial Intelligence Review**, 2021. Disponível em: https://doi.org/10.1007/s10462-021-09975-1
* **BREIMAN**, Leo. Random forests. **Machine Learning**, v. 45, p. 5–32, 2001.
* **CICEK**, Zeynep Idil Erzurum; **OZTURK**, Zehra Kamisli. Optimizing the artificial neural network parameters using a biased random key genetic algorithm for time series forecasting. **Applied Soft Computing Journal**, v. 102, 107091, 2021. Disponível em: https://www.sciencedirect.com/science/article/abs/pii/S1568494621000170
* **EL MRABET**, Zakaria et al. Random forest regressor-based approach for detecting fault location and duration in power systems. **Sensors**, v. 22, n. 2, p. 458, 2022. DOI: https://doi.org/10.3390/s22020458.
* **FAYYAD**, Usama; **STOLORZ**, Paul. Data mining and KDD: promise and challenges. **Future Generation Computer Systems**, v. 13, p. 99–115, 1997.
* **G.**, S.; **BRINDHA**, S. Hyperparameters optimization using Gridsearch Cross Validation Method for machine learning models in predicting diabetes mellitus risk. In: **INTERNATIONAL CONFERENCE ON COMMUNICATION, COMPUTING AND INTERNET OF THINGS (IC3IoT)**, 2022, Chennai, Índia. Anais [...]. IEEE, 2022.
* **LECUN**, Yann; **BENGIO**, Yoshua; **HINTON**, Geoffrey. Deep learning. **Nature**, v. 521, p. 436–444, 2015. DOI: https://doi.org/10.1038/nature14539.
* **MANTOVANI**, E. C. et al. Eficiência no uso da água de duas cultivares de batata-doce em resposta a diferentes lâminas de irrigação. **Horticultura Brasileira**, v. 31, n. 4, p. 602–606, 2013.
* **RICE**, Leslie; **WONG**, Eric; **KOLTER**, J. Zico. Overfitting in adversarially robust deep learning. 2020. Disponível em: https://doi.org/10.48550/arXiv.2002.11569
* **SOLTANOLKOTABI**, Mahdi; **JAVANMARD**, Adel; **LEE**, Jason D. Theoretical insights into the optimization landscape of over-parameterized shallow neural networks. **IEEE Transactions on Information Theory**, v. 65, n. 2, p. 742–769, fev. 2019. DOI: https://doi.org/10.1109/TIT.2018.2869182.
* **SUKAMTO**; **HADIYANTO**; **KURNIANINGSIH**. KNN optimization using Grid Search algorithm for preeclampsia imbalance class. **E3S Web of Conferences**, v. 448, 02057, 2023. DOI: https://doi.org/10.1051/e3sconf/202344802057.