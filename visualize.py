# visualize 封装成为函数 ， 或者是使用os命令来执行该文件
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns


# normally display chinese character
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False



data = pd.read_excel('（全部白夜班）final_ori_data.xlsx')
print(data.head(),data.columns)

# List of selected variables
selected_variables = ['里程 km', '平均温度(°F)', '平均风速(knots)', '最高气温(°F)', '最低气温(°F)', '降水量(in)']

# Set up the matplotlib figure
plt.figure(figsize=(24, 14))

# For each variable, plot a boxplot and scatterplot/regression plot
for i, var in enumerate(selected_variables, 1):
    plt.subplot(2, len(selected_variables), i)
    sns.boxplot(data[var])
    plt.title(f"Boxplot for {var}")

    plt.subplot(2, len(selected_variables), i + len(selected_variables))
    sns.regplot(x=data[var], y=data['吨公里油耗 L/km/t'], scatter_kws={'s':10}, line_kws={'color':'red'})
    plt.title(f"{var} vs 吨公里油耗 L/km/t")
plt.savefig('1.jpg',dpi=300)
plt.tight_layout()
plt.show()

## barplot
# List of selected categorical variables
categorical_variables = ['车型', '道路质量', '班次']

# Set up the matplotlib figure
plt.figure(figsize=(15, 5))

# For each categorical variable, plot a bar chart
for i, var in enumerate(categorical_variables, 1):
    plt.subplot(1, len(categorical_variables), i)
    sns.countplot(x=var, data=data)
    plt.title(f"Distribution of {var}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


## heatmap
# Calculate the correlation matrix
data = data.drop(columns=['Unnamed: 0'],axis=1)
correlation_matrix = data.select_dtypes(include=['number']).corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 9))

# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmax=1, vmin=-1, linewidths=0.5, linecolor='white')
plt.title("Correlation Heatmap")
plt.show()

conda activate your_env_name





