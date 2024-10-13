import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

file_path = 'stock_data_2018.csv'
valid_tickers = []
data = []
try:
    data = pd.read_csv(file_path)
    valid_tickers = data['Ticker'].unique().tolist()
except FileNotFoundError:
    print(f"Файл {file_path} не найден. Проверьте путь к файлу.")
except pd.errors.EmptyDataError:
    print(f"Файл {file_path} пуст.")
except Exception as e:
    print(f"Произошла ошибка: {e}")

if data.empty:
    print("Файл данных пуст. Невозможно продолжить.")
    exit()

E_dict = {}
Sigma_dict = {}

for ticker in valid_tickers:
    log_returns = data[data['Ticker'] == ticker]['log_return']
    E_dict[ticker] = log_returns.mean()
    Sigma_dict[ticker] = log_returns.std()

# Вывод результатов
#print("Valid Tickers:", valid_tickers)
#print("E Dictionary:", E_dict)
#print("Sigma Dictionary:", Sigma_dict)

pareto_optimal_assets = []

assets = list(E_dict.keys())
E_values = np.array(list(E_dict.values()))
Sigma_values = np.array(list(Sigma_dict.values()))

for i in range(len(assets)):
    current_E = E_values[i]
    current_Sigma = Sigma_values[i]
    is_optimal = True
    for j in range(len(assets)):
        if i != j:
            if (E_values[j] >= current_E and Sigma_values[j] <= current_Sigma):
                is_optimal = False
                break
    if is_optimal:
        pareto_optimal_assets.append(assets[i])

#print("Парето-оптимальные активы:", pareto_optimal_assets)
#print("Всего", len(pareto_optimal_assets), "Парето-оптимальных активов")


z_scores = np.abs(stats.zscore(Sigma_values))
threshold = 3
filtered_Sigma_values = Sigma_values[(z_scores < threshold)]
filtered_E_values = E_values[(z_scores < threshold)]
filtered_assets = np.array(assets)[(z_scores < threshold)]

correlation_matrix = data.pivot_table(values='log_return', index='Date', columns='Ticker').corr()

portfolio_list = []

portfolio_list.extend(pareto_optimal_assets)

added_assets = 0
for asset in pareto_optimal_assets:
    correlations = correlation_matrix[asset]
    sorted_correlations = correlations.sort_values(ascending=True)
    for ticker in sorted_correlations.index:
        if ticker not in portfolio_list and ticker != asset and E_dict[ticker] > 0:
            portfolio_list.append(ticker)
            added_assets += 1
            if added_assets == 15:
                break
    if added_assets == 15:
        break

if added_assets < 15:
    for asset in pareto_optimal_assets:
        correlations = correlation_matrix[asset]
        sorted_correlations = correlations.sort_values(ascending=False)
        for ticker in sorted_correlations.index:
            if ticker not in portfolio_list and ticker != asset and E_dict[ticker] > 0:
                portfolio_list.append(ticker)
                added_assets += 1
                if added_assets == 15:
                    break
        if added_assets == 15:
            break

if len(portfolio_list) != 50:
    print("Не удалось сформировать набор из 50 активов.")
    if len(portfolio_list) > 0:
        print("В наборе имеется", len(portfolio_list), "акций.")
        print("Портфель инвестиций: ", portfolio_list)
else:
    print("Тикеры компаний, входящих в набор из 50 активов:", portfolio_list)

n_min_risk_assets = 10
weights = np.zeros(len(portfolio_list))

returns = np.zeros(len(portfolio_list))
for i, ticker in enumerate(portfolio_list):
    weights[i] = 1 / len(portfolio_list)
    returns[i] = E_dict[ticker]

portfolio_df = pd.DataFrame({'Tickers': portfolio_list})
portfolio_df['Sigma'] = portfolio_df['Tickers'].map(Sigma_dict)
portfolio_df['Sigma_squared'] = [x*x for x in portfolio_df['Sigma']]


def portfolio_variance(weights_):
    cov_matrix = portfolio_df[['Sigma_squared']].to_numpy()
    portfolio_variance_ = weights_.T @ cov_matrix
    portfolio_variance_ = np.sum(portfolio_variance_ * weights_)
    return portfolio_variance_


constraints_with_short_sales = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
)


constraints_without_short_sales = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    {'type': 'ineq', 'fun': lambda x: x},
)

result_without_short_sales = minimize(portfolio_variance, weights, method='SLSQP', constraints=constraints_without_short_sales)
optimal_weights = result_without_short_sales.x
top_10_assets_no_short_sales = np.argsort(optimal_weights)[-n_min_risk_assets:]
print(f"Топ {n_min_risk_assets} активов с минимальным риском (короткие продажи запрещёны):")
plt.figure(figsize=(10, 6))
for i in top_10_assets_no_short_sales:
    print(portfolio_list[i])
    plt.scatter(E_dict[portfolio_list[i]], Sigma_dict[portfolio_list[i]], color='orange', s=100)
    plt.annotate(portfolio_list[i], (E_dict[portfolio_list[i]], Sigma_dict[portfolio_list[i]]), fontsize=8)
plt.scatter(E_dict['KNIP11.SA'], Sigma_dict['KNIP11.SA'], color='orange',
            s=100, label='Активы с минимальным риском (короткие продажи запрещены)')
plt.xlabel('Sigma (Среднеквадратическое отклонение)')
plt.ylabel('E (Математическое ожидание)')
plt.title('Карта активов')
plt.legend()
plt.grid(True)

result_with_short_sales = minimize(portfolio_variance, weights, method='SLSQP', constraints=constraints_with_short_sales)
optimal_weights = result_with_short_sales.x
top_10_assets_no_short_sales = np.argsort(optimal_weights)[-n_min_risk_assets:]
print(f"Топ {n_min_risk_assets} активов с минимальным риском (короткие продажи разрешены):")
plt.figure(figsize=(10, 6))
for i in top_10_assets_no_short_sales:
    print(portfolio_list[i])
    plt.scatter(E_dict[portfolio_list[i]], Sigma_dict[portfolio_list[i]], color='orange', s=100)
    plt.annotate(portfolio_list[i], (E_dict[portfolio_list[i]], Sigma_dict[portfolio_list[i]]), fontsize=8)
plt.scatter(E_dict['FRIO3.SA'], Sigma_dict['FRIO3.SA'], color='orange',
            s=100, label='Активы с минимальным риском (короткие продажи разрешены)')
plt.xlabel('Sigma (Среднеквадратическое отклонение)')
plt.ylabel('E (Математическое ожидание)')
plt.title('Карта активов')
plt.legend()
plt.grid(True)
plt.show()

