import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')

class ConsumoAguaLavoura:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_encoded = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.rf_model = None
        self.nn_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def carregar_planilha(self):
        try:
            self.df = pd.read_excel(self.file_path)
            print("Dados carregados.")
            print(f"Shape dos dados: {self.df.shape}")
            print(f"Colunas: {list(self.df.columns)}")
            print(f"Valores nulos: {self.df.isnull().sum().sum()}")
            return True
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False
    
    def analise_descritiva(self):
        print("\n" + "="*60)
        print("ANÁLISE DESCRITIVA DO CONSUMO DE ÁGUA")
        print("="*60)
        
        print("\nEstatísticas Descritivas:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe())
        
        self.create_plots_descritivos()
        
        self.correlacao()
        
    def create_plots_descritivos(self):
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.hist(self.df['Water_Usage(cubic meters)'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribuição do Consumo de Água', fontsize=12, fontweight='bold')
        plt.xlabel('Consumo de Água (m³)')
        plt.ylabel('Frequência')

        plt.subplot(2, 3, 2)
        sns.boxplot(data=self.df, x='Crop_Type', y='Water_Usage(cubic meters)')
        plt.title('Consumo de Água por Tipo de Cultura', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 3)
        irrigation_water = self.df.groupby('Irrigation_Type')['Water_Usage(cubic meters)'].mean()
        bars = plt.bar(irrigation_water.index, irrigation_water.values, color='lightgreen', alpha=0.8)
        plt.title('Consumo Médio por Tipo de Irrigação', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Consumo Médio (m³)')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.subplot(2, 3, 4)
        season_water = self.df.groupby('Season')['Water_Usage(cubic meters)'].mean()
        colors = ['gold', 'lightcoral', 'lightblue', 'lightgreen']
        plt.pie(season_water.values, labels=season_water.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Distribuição do Consumo por Estação', fontsize=12, fontweight='bold')
        
        plt.subplot(2, 3, 5)
        plt.scatter(self.df['Farm_Area(acres)'], self.df['Water_Usage(cubic meters)'], 
                   alpha=0.6, color='purple')
        plt.title('Área da Fazenda vs Consumo de Água', fontsize=12, fontweight='bold')
        plt.xlabel('Área da Fazenda (acres)')
        plt.ylabel('Consumo de Água (m³)')
        
        plt.subplot(2, 3, 6)
        soil_water = self.df.groupby('Soil_Type')['Water_Usage(cubic meters)'].mean().sort_values()
        bars = plt.barh(soil_water.index, soil_water.values, color='orange', alpha=0.7)
        plt.title('Consumo Médio por Tipo de Solo', fontsize=12, fontweight='bold')
        plt.xlabel('Consumo Médio (m³)')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.0f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('analise_descritiva_consumo_agua.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Gráfico salvo: 'analise_descritiva_consumo_agua.png'")
        
    def correlacao(self):

        df_temp = self.df.copy()

        if 'Farm_ID' in df_temp.columns:
            df_temp = df_temp.drop('Farm_ID', axis=1)
        
        categorical_cols = ['Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season']
        
        for col in categorical_cols:
            if col in df_temp.columns:
                le = LabelEncoder()
                df_temp[col] = le.fit_transform(df_temp[col])
        
        plt.figure(figsize=(12, 8))
        correlation_matrix = df_temp.corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Matriz de Correlação - Todas as Variáveis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('matriz_correlacao.png', dpi=300, bbox_inches='tight')
        plt.show()

        water_corr = correlation_matrix['Water_Usage(cubic meters)'].abs().sort_values(ascending=False)
        print(f"\n Correlações com Consumo de Água:")
        for var, corr in water_corr.items():
            if var != 'Water_Usage(cubic meters)':
                print(f"  {var}: {corr:.3f}")
        
        print("Gráfico salvo: 'matriz_correlacao.png'")
    
    def preprocess_data(self):
        print("\n" + "="*60)
        print("PRÉ-PROCESSAMENTO DOS DADOS")
        print("="*60)

        self.df_encoded = self.df.copy()

        categorical_cols = ['Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_encoded[col] = le.fit_transform(self.df_encoded[col])
            self.label_encoders[col] = le
            print(f"{col} codificado")
        
        feature_cols = [col for col in self.df_encoded.columns 
                       if col not in ['Water_Usage(cubic meters)', 'Farm_ID']]
        X = self.df_encoded[feature_cols]
        y = self.df_encoded['Water_Usage(cubic meters)']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Dados de treino: {self.X_train.shape}")
        print(f"Dados de teste: {self.X_test.shape}")
        print("Pré-processamento concluído!")
        
    def treinamento_random_forest(self):
        print("\n Treinando Random Forest")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        self.rf_model = grid_search.best_estimator_
        
        print(f"Melhores parâmetros RF: {grid_search.best_params_}")
        
        cv_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, 
                                   cv=5, scoring='neg_mean_squared_error')
        print(f"CV Score RF: {-cv_scores.mean():.2f} (±{cv_scores.std()*2:.2f})")
        
    def treinamento_rede_neural(self):
        print("\n Treinando Rede Neural")
        
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'activation': ['relu', 'tanh'],
            'learning_rate': ['constant', 'adaptive']
        }
        
        nn = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
        
        grid_search = GridSearchCV(
            nn, param_grid, cv=5, scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        self.nn_model = grid_search.best_estimator_
        
        print(f"Melhores parâmetros NN: {grid_search.best_params_}")

        cv_scores = cross_val_score(self.nn_model, self.X_train_scaled, self.y_train, 
                                   cv=5, scoring='neg_mean_squared_error')
        print(f"CV Score NN: {-cv_scores.mean():.2f} (±{cv_scores.std()*2:.2f})")
    
    def evaluate_models(self):
        print("\n" + "="*60)
        print("AVALIAÇÃO DOS MODELOS")
        print("="*60)

        rf_pred = self.rf_model.predict(self.X_test)
        nn_pred = self.nn_model.predict(self.X_test_scaled)

        models_metricas = {
            'Random Forest': {
                'MSE': mean_squared_error(self.y_test, rf_pred),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, rf_pred)),
                'MAE': mean_absolute_error(self.y_test, rf_pred),
                'R²': r2_score(self.y_test, rf_pred)
            },
            'Neural Network': {
                'MSE': mean_squared_error(self.y_test, nn_pred),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, nn_pred)),
                'MAE': mean_absolute_error(self.y_test, nn_pred),
                'R²': r2_score(self.y_test, nn_pred)
            }
        }

        print("\nMÉTRICAS DE PERFORMANCE:")
        print("-" * 50)
        for model_name, metricas in models_metricas.items():
            print(f"\n{model_name}:")
            for metric_name, value in metricas.items():
                print(f"  {metric_name}: {value:.4f}")
        
        self.plots_metricas(rf_pred, nn_pred)
        
        self.plot_importancia()
        
        return models_metricas
    
    def plots_metricas(self, rf_pred, nn_pred):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].scatter(self.y_test, rf_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Valores Reais')
        axes[0, 0].set_ylabel('Predições')
        axes[0, 0].set_title('Random Forest: Predito vs Real')
        
        axes[0, 1].scatter(self.y_test, nn_pred, alpha=0.6, color='green')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Valores Reais')
        axes[0, 1].set_ylabel('Predições')
        axes[0, 1].set_title('Neural Network: Predito vs Real')
        
        axes[0, 2].scatter(rf_pred, nn_pred, alpha=0.6, color='purple')
        axes[0, 2].plot([min(rf_pred.min(), nn_pred.min()), max(rf_pred.max(), nn_pred.max())], 
                       [min(rf_pred.min(), nn_pred.min()), max(rf_pred.max(), nn_pred.max())], 
                       'r--', lw=2)
        axes[0, 2].set_xlabel('Random Forest')
        axes[0, 2].set_ylabel('Neural Network')
        axes[0, 2].set_title('Comparação entre Modelos')
        
        rf_residuals = self.y_test - rf_pred
        nn_residuals = self.y_test - nn_pred
        
        axes[1, 0].hist(rf_residuals, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_xlabel('Resíduos')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Random Forest: Distribuição dos Resíduos')
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        
        axes[1, 1].hist(nn_residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Resíduos')
        axes[1, 1].set_ylabel('Frequência')
        axes[1, 1].set_title('Neural Network: Distribuição dos Resíduos')
        axes[1, 1].axvline(x=0, color='red', linestyle='--')
        
        metricas_rf = [
            np.sqrt(mean_squared_error(self.y_test, rf_pred)),
            mean_absolute_error(self.y_test, rf_pred),
            r2_score(self.y_test, rf_pred)
        ]
        
        metricas_nn = [
            np.sqrt(mean_squared_error(self.y_test, nn_pred)),
            mean_absolute_error(self.y_test, nn_pred),
            r2_score(self.y_test, nn_pred)
        ]
        
        metric_names = ['RMSE', 'MAE', 'R²']
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, metricas_rf, width, label='Random Forest', alpha=0.8)
        axes[1, 2].bar(x + width/2, metricas_nn, width, label='Neural Network', alpha=0.8)
        axes[1, 2].set_xlabel('Métricas')
        axes[1, 2].set_ylabel('Valores')
        axes[1, 2].set_title('Comparação de Métricas')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metric_names)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('avaliacao_modelos.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Gráfico salvo: 'avaliacao_modelos.png'")
    
    def plot_importancia(self):
        feature_names = [col for col in self.df_encoded.columns 
                        if col not in ['Water_Usage(cubic meters)', 'Farm_ID']]
        importance = self.rf_model.feature_importances_
        
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(importance)), importance[indices], alpha=0.8, color='skyblue')
        plt.title('Importância das Features - Random Forest', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importância')
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('importancia.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Gráfico salvo: 'importancia.png'")
        
        print("\nIMPORTÂNCIA DAS FEATURES:")
        print("-" * 30)
        for i in indices:
            print(f"{feature_names[i]}: {importance[i]:.4f}")
    
    def predicao_consumo_agua(self, farm_data):
        print("\n" + "="*60)
        print("PREDIÇÃO DE CONSUMO DE ÁGUA")
        print("="*60)

        df_new = pd.DataFrame([{k: v for k, v in farm_data.items() if k != 'Farm_ID'}])
        
        for col, encoder in self.label_encoders.items():
            if col in df_new.columns:
                df_new[col] = encoder.transform(df_new[col])
        
        df_new_scaled = self.scaler.transform(df_new)

        rf_prediction = self.rf_model.predict(df_new)[0]
        nn_prediction = self.nn_model.predict(df_new_scaled)[0]
        
        print(f"Random Forest: {rf_prediction:.2f} m³")
        print(f"Neural Network: {nn_prediction:.2f} m³")
        print(f"Média dos modelos: {(rf_prediction + nn_prediction)/2:.2f} m³")
        
        return rf_prediction, nn_prediction
    
    def analise(self):
        print("🚀 INICIANDO ANÁLISE COMPLETA DE CONSUMO DE ÁGUA")
        print("="*70)

        if not self.carregar_planilha():
            return
        
        self.analise_descritiva()
        
        self.preprocess_data()
        
        self.treinamento_random_forest()
        self.treinamento_rede_neural()
        
        metricas = self.evaluate_models()
        
        print("\n" + "="*70)
        print("Gráficos salvos:")
        print("   - analise_descritiva_consumo_agua.png")
        print("   - matriz_correlacao.png")
        print("   - avaliacao_modelos.png")
        print("   - importancia.png")
        print("="*70)
        
        return metricas

if __name__ == "__main__":

    analyzer = ConsumoAguaLavoura('irrigacao.xlsx')

    metricas = analyzer.analise()

