from dash import Dash, dcc, html
import pandas as pd
import requests
import plotly.express as px
from dash.dependencies import Input, Output
from plotly import graph_objects as go
import plotly.figure_factory as ff
from plotly.figure_factory import create_annotated_heatmap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm

# url = "https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2024/April/WEOApr2024all.ashx"
filename_xls = "WEOApr2024all.xls"

# response = requests.get(url)
# if response.status_code == 200:
#    with open(filename_xls, 'wb') as f:
#        f.write(response.content)
#    print(f"Файл сохранен как {filename_xls}")
# else:
#    print(f"Ошибка загрузки файла. Статус: {response.status_code}")

df = pd.read_csv(filename_xls, sep='\t', encoding='utf-16-le')

keywords = [
   'NGDPD', 'NGDP_D', 'NGDPDPC', 'NID_NGDP', 'NGSD_NGDP', 'PCPI', 'TM_RPCH', 'TX_RPCH',
   'LUR', 'LE', 'LP', 'GGR', 'GGX', 'GGXCNL', 'GGSB', 'GGXONLB', 'GGXWDN', 'GGXWDG', 'BCA'
]

filtered_df = df[df.apply(lambda row: any(keyword in str(row).upper() for keyword in keywords), axis=1)]
filtered_df.to_excel('filtered_file.xlsx', index=False)
print("Фильтрация завершена, отфильтрованный файл сохранен как 'filtered_file.xlsx'.")

df = pd.read_excel('filtered_file.xlsx')

indicators = ['NGDPDPC', 'NID_NGDP', 'NGSD_NGDP', 'PCPI', 'TM_RPCH', 'TX_RPCH',
    'LUR', 'LE', 'LP', 'GGR', 'GGX', 'GGXCNL', 'GGSB', 'GGXONLB', 'GGXWDN', 'GGXWDG', 'BCA']

years = [str(year) for year in range(2010, 2023)]

df_melted = df.melt(id_vars=['Country', 'ISO', 'WEO Subject Code'],
                    value_vars=years,
                    var_name='Year', value_name='Value')

#фильтрация по индикаторам
df_filtered = df_melted[df_melted['WEO Subject Code'].isin(indicators)]

#преобразование в широкий формат
df_wide = df_filtered.pivot_table(index=['Country', 'ISO', 'Year'],
                                  columns='WEO Subject Code',
                                  values='Value',
                                 aggfunc='first').reset_index()

#преобразование данных в числовой формат
for indicator in indicators:
    if indicator in df_wide.columns:
        df_wide[indicator] = df_wide[indicator].astype(str).str.replace(',', '', regex=True)
        df_wide[indicator] = pd.to_numeric(df_wide[indicator], errors='coerce')

df_wide.to_excel('filtered_file2.xlsx', index=False)
print("Файл 'filtered_file2.xlsx' успешно сохранен.")

df_raw = pd.read_excel('filtered_file2.xlsx', engine='openpyxl')

#преобразуем данные в длинный формат
df_melted = df_raw.melt(id_vars=["Country", "ISO", "Year"], var_name="Indicator", value_name="Value")

#список показателей с их корректными описаниями и комментариями
indicators = {
    "NGDPDPC": {"name": "ВВП на душу населения (текущие цены)", "comment": "Средний уровень экономической активности или дохода на человека."},
    "PCPI": {"name": "Индекс потребительских цен (CPI)", "comment": "Измеряет изменение стоимости корзины потребительских товаров и услуг во времени."},
    "TM_RPCH": {"name": "Рост импорта (процентное изменение)", "comment": "Отражает темпы роста импорта товаров и услуг по сравнению с предыдущим годом."},
    "TX_RPCH": {"name": "Рост экспорта (процентное изменение)", "comment": "Отображает изменение объема экспорта товаров и услуг в реальном выражении."},
    "LP": {"name": "Численность населения (млн человек)", "comment": "Общее количество людей, проживающих в стране."},
    "LUR": {"name": "Уровень безработицы (%)", "comment": "Процентное соотношение безработных к численности рабочей силы."},
    "LE": {"name": "Общая занятость (млн человек)", "comment": "Общее количество занятых в экономике страны."},
    "NID_NGDP": {"name": "Инвестиции (процент от ВВП)", "comment": "Общий объем валовых инвестиций по отношению к ВВП."},
    "NGSD_NGDP": {"name": "Национальные сбережения (процент от ВВП)", "comment": "Доля валовых национальных сбережений в ВВП."},
    "GGR": {"name": "Госдоходы (млрд нац. валюты)", "comment": "Объем доходов государственного сектора, включая налоги и социальные взносы."},
    "GGX": {"name": "Госрасходы (млрд нац. валюты)", "comment": "Общий объем государственных расходов, включая социальные программы и инвестиции."},
    "GGXCNL": {"name": "Гос. чистое кредитование/заимствование (%)", "comment": "Разница между доходами и расходами госуправления по отношению к ВВП."},
    "GGSB": {"name": "Структурный баланс гос. сектора (млрд нац. валюты)", "comment": "Корректированный баланс бюджета с учетом экономического цикла."},
    "BCA": {"name": "Баланс текущего счета (млрд USD)", "comment": "Разница между экспортом и импортом товаров, услуг и финансовых потоков."}
}

groups = {
    "Макроэкономика и рост": ["NGDPDPC", "PCPI", "TM_RPCH", "TX_RPCH"],
    "Рынок труда и население": ["LP", "LE", "LUR"],
    "Финансовая устойчивость и гос. сектор": ["NID_NGDP", "NGSD_NGDP", "GGR", "GGX", "GGXCNL", "GGSB", "BCA"],
    "Аналитика": []
}

latest_year = df_melted['Year'].max()
def get_top10(ind):
    temp = df_melted[(df_melted['Indicator'] == ind) & (df_melted['Year'] == latest_year)]
    return temp.sort_values(by='Value', ascending=False).head(10)['Country'].unique().tolist()

countries = sorted(df_melted['Country'].dropna().unique())

for indicator in indicators:
    if indicator in indicators:
        z_scores = ((df_raw.select_dtypes(include='number') - df_raw.select_dtypes(include='number').mean()) / df_raw.select_dtypes(include='number').std()).abs()
        outliers = (z_scores > 3).sum().reset_index()
        outliers.columns = ['Indicator', 'OutlierCount']

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Экономический дашборд МВФ (WEO 2024)", style={"textAlign": "center"}),
    dcc.Tabs([
        dcc.Tab(label=group_name, children=[
            html.Div([
                *[html.Div([
                    html.H1(f"{indicators[ind]['name']}", style={"margin-top": "20px"}),
                    html.P(indicators[ind]['comment'], style={"font-style": "italic"}),
                    html.Label(f"Выберите страны для {indicators[ind]['name']}:"),
                    dcc.Dropdown(
                        id=f"dropdown-{ind}",
                        options=[{"label": country, "value": country} for country in sorted(df_melted['Country'].unique())],
                        value=get_top10(ind),
                        multi=True
                    ),
                    dcc.Graph(id=f"graph-{ind}")
                ]) for ind in groups[group_name]]
            ])
        ]) if group_name != "Аналитика" else dcc.Tab(label="Аналитика", children=[
            html.Div([
                html.H1("1. Корреляционная матрица по странам", style={"margin-top": "20px"}),
                html.Label("Выберите страну для анализа (или выберите 'Все страны')"),
                dcc.Dropdown(
                    id='corr-country-dropdown', 
                    options=[{"label": c, "value": c} for c in countries] + [{"label": "Все страны", "value": "all"}],
                    value="all",
                    multi=False
                ),
                dcc.Graph(id='corr-matrix-graph'),  
                html.Div(id='top-correlated') 
            ]),
            html.Div([
                html.H1("2. Коэффициент вариации (CV)", style={"margin-top": "20px"}),
                html.P("Коэффициент вариации (CV) для каждого показателя:"),
                html.Div(id="cv-values")
            ]),
            html.Div([
                html.H1("3. Кластеризация стран (PCA + KMeans)", style={"margin-top": "20px"}),
                dcc.Graph(id="pca-kmeans")
            ]),
            html.Div([
                dcc.Input(id='dummy', value='init', type='hidden'),  
                html.H1("4. Тепловая карта пропущенных значения"),
                dcc.Graph(id='missing-heatmap')
            ]),
            html.Div([
                # Тесты на стационарность
                html.H1("5. Тесты на стационарность (ADF, KPSS)", style={"margin-top": "20px"}),
                html.H2("Стационарный ряд имеет постоянные статистические характеристики (среднее, дисперсия) со временем, в то время как нестационарный ряд изменяется во времени, что делает его более сложным для анализа и прогнозирования."),
                html.H3("ADF и KPSS (единичные корни Дики-Фуллера (ADF) и Квятковского, Филлипса, Шмидта и Шина (KPSS))— это тесты для проверки стационарности временных рядов, важные для экономического анализа. ADF проверяет наличие единичного корня (нулевая гипотеза — ряд нестационарен), и если p-value < 0.05, ряд считается стационарным. KPSS, наоборот, тестирует гипотезу о стационарности (нулевая — ряд стационарен), и если p-value < 0.05, ряд нестационарен. Их совместное применение помогает точно определить, нужны ли дифференцирование или устранение тренда перед моделированием"),
                html.Div(id="stationarity-result")
            ]),
            html.Div([
                html.H1("6. Выбросы по оценкам (IQR межквартильный размах)"),
                dcc.Graph(figure=px.bar(outliers, x='Indicator', y='OutlierCount', title='Количество выбросов по показателям'))
            ]),
            html.Div([
                html.H3('Интерактивная карта экономических показателей'),
                dcc.Graph(
                    figure=px.choropleth(
                        df_wide,
                        locations='ISO',
                        locationmode='ISO-3',
                        color='NGDPDPC',
                        hover_name='Country',
                        hover_data={
                            'NGDPDPC': True,
                            'NID_NGDP': True,
                            'NGSD_NGDP': True,
                            'PCPI': True,
                            'TM_RPCH': True,
                            'TX_RPCH': True,
                            'LUR': True,
                            'LE': True,
                            'LP': True,
                            'GGR': True,
                            'GGX': True,
                            'GGXCNL': True,
                            'GGSB': True,
                            'GGXONLB': True,
                            'GGXWDN': True,
                            'GGXWDG': True,
                            'BCA': True
                        },
                        color_continuous_scale=['white', 'green'],
                        animation_frame='Year',
                        projection='natural earth'
                    ).update_layout(
                        geo=dict(showframe=False, showcoastlines=False),
                        showlegend=False
                    )
                )
            ])    

        ]) for group_name in groups
    ])
])

for group_name in groups:
    for ind in groups[group_name]:
        @app.callback(
            Output(f"graph-{ind}", "figure"),
            Input(f"dropdown-{ind}", "value")
        )
        def update_graph(countries, ind=ind):
            df_plot = df_melted[(df_melted['Country'].isin(countries)) & (df_melted['Indicator'] == ind)]
            fig = px.line(df_plot, x="Year", y="Value", color="Country", title=indicators[ind]['name'])
            return fig

@app.callback(
    [Output("corr-matrix-graph", "figure"),
     Output("top-correlated", "children")],
    Input("corr-country-dropdown", "value")
)
def update_corr_matrix(selected_country):
    valid_indicators = list(indicators.keys())

    if selected_country == "all":
        df_filtered = df_melted[df_melted['Indicator'].isin(valid_indicators)]
    else:
        df_filtered = df_melted[(df_melted['Country'] == selected_country) & 
                                (df_melted['Indicator'].isin(valid_indicators))]
        
    df_pivot = df_filtered.pivot_table(index="Year", columns="Indicator", values="Value")

    df_pivot = df_pivot.dropna(axis=1, thresh=int(len(df_pivot) * 0.6))
    corr_matrix = df_pivot.corr()

    corr_copy = corr_matrix.copy()
    np.fill_diagonal(corr_copy.values, np.nan)

    corr_pairs = corr_copy.abs().unstack().dropna()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs = corr_pairs.sort_values(ascending=False)

    if not corr_pairs.empty:
        ind1, ind2 = corr_pairs.index[0]
        corr_value = corr_matrix.loc[ind1, ind2]

        name1 = indicators[ind1]['name']
        name2 = indicators[ind2]['name']
        comment1 = indicators[ind1]['comment']
        comment2 = indicators[ind2]['comment']

        top_text = html.Div([
            html.H4("Наиболее скоррелированные показатели:"),
            html.P(f"{name1} ({comment1}) ↔ {name2} ({comment2}) — корреляция: {corr_value:.2f}")
        ])
    else:
        top_text = html.Div([html.H4("Недостаточно данных для оценки корреляции.")])

    text_values = np.round(corr_matrix.values, 2).astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[indicators[i]['name'] for i in corr_matrix.columns],
        y=[indicators[i]['name'] for i in corr_matrix.index],
        colorscale='RdYlGn',
        zmin=-1, zmax=1,
        colorbar=dict(title="Корреляция"),
        text=text_values, 
        hoverinfo='text',  
        showscale=True,
        texttemplate="%{text}",  
        textfont=dict(size=12, color="black") 
    ))

    fig.update_layout(
        title="Матрица корреляции показателей",
        xaxis_nticks=len(corr_matrix.columns),
        yaxis_nticks=len(corr_matrix.index),
        height=800
    )

    return fig, top_text 

@app.callback(
    Output('cv-values', 'children'), 
    [Input('corr-country-dropdown', 'value')] 
)
def cv_analysis(selected_country):  
    df_filtered = df_melted 

    cv_results = {}
    for indicator in indicators:
        indicator_data = df_filtered[df_filtered['Indicator'] == indicator]
        cv = indicator_data.groupby('Indicator')['Value'].std() / indicator_data.groupby('Indicator')['Value'].mean()
        cv_results[indicator] = cv[0]
    
    fig = px.bar(
        x=[indicators[indicator]['name'] for indicator in indicators],  
        y=list(cv_results.values()),  
        labels={'x': 'Индикаторы', 'y': 'Коэффициент вариации (CV)'},
        title="Коэффициент вариации (CV) для каждого показателя"
    )
    
    return dcc.Graph(figure=fig)  

from sklearn.impute import SimpleImputer

@app.callback(
    Output('pca-kmeans', 'figure'),
    [Input('corr-country-dropdown', 'value')]  
)
def pca_kmeans_analysis(selected_country):
    df_filtered = df_melted 
    df_pivot = df_filtered.pivot_table(index='Country', columns='Indicator', values='Value')

    df_pivot = df_pivot.dropna(axis=1, thresh=int(len(df_pivot) * 0.6))

    imputer = SimpleImputer(strategy='mean') 
    df_imputed = pd.DataFrame(imputer.fit_transform(df_pivot), columns=df_pivot.columns, index=df_pivot.index)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns, index=df_imputed.index)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled)

    fig = px.scatter(
        x=pca_components[:, 0], y=pca_components[:, 1],
        color=df_scaled['cluster'].astype(str),
        text=df_scaled.index,
        labels={"x": "PCA Component 1", "y": "PCA Component 2"},
        title="Кластеризация стран с использованием PCA + KMeans"
    )
    return fig

@app.callback(
    Output('missing-heatmap', 'figure'),
    [Input('dummy', 'value')] 
)
def plot_missing_treemap(_):
    missing_data = df_melted[df_melted['Value'].isnull()]
    missing_count_per_country = missing_data.groupby('Country').size().reset_index(name='missing_count')

    print(f"Максимальное количество пропусков: {missing_count_per_country['missing_count'].max()}")

    fig = px.treemap(
        missing_count_per_country,
        path=['Country'],
        values='missing_count',  
        color='missing_count',  
        color_continuous_scale='Reds',  
        title="Пропуски по странам по всем годам и показателям"
    )

    fig.update_layout(height=600)  

    return fig

def adf_test(series):
    series = series.replace([float('inf'), -float('inf')], pd.NA)
    series = series.dropna() 
    if len(series) < 2 or series.nunique() == 1:  
        return None 
    try:
        result = adfuller(series)
        p_value = result[1]
        
        return p_value  
    except Exception as e:
        return None 

def kpss_test(series):

    series = series.replace([float('inf'), -float('inf')], pd.NA)
    series = series.dropna() 
    if len(series) < 2 or series.nunique() == 1:
        return None 
    try:
        result = kpss(series, regression='c', nlags="auto")
        p_value = result[1]
        
        return p_value  
    except Exception as e:
        return None  


@app.callback(
    Output('stationarity-result', 'children'),
    Input('corr-country-dropdown', 'value')  
)
def stationarity_tests(selected_country):
    
    valid_indicators = list(indicators.keys())
    if selected_country == "all":
        df_filtered = df_melted[df_melted['Indicator'].isin(valid_indicators)]
    else:
        df_filtered = df_melted[(df_melted['Country'] == selected_country) & 
                                (df_melted['Indicator'].isin(valid_indicators))]
    
    if df_filtered.empty:
        return "Нет данных для выбранной страны"
    
    results = []

    for indicator in df_filtered['Indicator'].unique():
        indicator_data = df_filtered[df_filtered['Indicator'] == indicator]

        adf_p_value = adf_test(indicator_data['Value'])
        if adf_p_value is not None and isinstance(adf_p_value, float):
            adf_result = "Стационарен" if adf_p_value < 0.05 else "Не стационарен"
        else:
            adf_result = "Не удалось выполнить тест"

        kpss_p_value = kpss_test(indicator_data['Value'])
        if kpss_p_value is not None and isinstance(kpss_p_value, float):
            kpss_result = "Стационарен" if kpss_p_value > 0.05 else "Не стационарен"  
        else:
            kpss_result = "Не удалось выполнить тест"

        indicator_name = indicators[indicator]['name'] if indicator in indicators else indicator
        results.append(f"{selected_country} - {indicator_name}: ADF p-value = {adf_p_value} ({adf_result}), KPSS p-value = {kpss_p_value} ({kpss_result})")

    return html.Div([html.P(result) for result in results])

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=10000)

