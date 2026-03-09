## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy.stats import boxcox 


data = pd.read_csv('Data_to_Transform.csv')

print("Original Dataset:")
print(data.head())

Original Dataset:
   Moderate Positive Skew  Highly Positive Skew  Moderate Negative Skew  \
0                0.899990              2.895074               11.180748   
1                1.113554              2.962385               10.842938   
2                1.156830              2.966378               10.817934   
3                1.264131              3.000324               10.764570   
4                1.323914              3.012109               10.753117   

   Highly Negative Skew  
0              9.027485  
1              9.009762  
2              9.006134  
3              9.000125  
4              8.981296  


data.fillna(data.mean(numeric_only=True), inplace=True)

	Moderate Positive Skew	Highly Positive Skew	Moderate Negative Skew	Highly Negative Skew
0	0.899990	2.895074	11.180748	9.027485
1	1.113554	2.962385	10.842938	9.009762
2	1.156830	2.966378	10.817934	9.006134
3	1.264131	3.000324	10.764570	9.000125
4	1.323914	3.012109	10.753117	8.981296
...	...	...	...	...
9995	14.749050	16.289513	-2.980821	-3.254882
9996	14.854474	16.396252	-3.147526	-3.772332
9997	15.262103	17.102991	-3.517256	-4.717950
9998	15.269983	17.628467	-4.689833	-5.670496
9999	16.204517	18.052331	-6.335679	-7.036091
10000 rows × 4 columns


numeric_column = data.select_dtypes(include=np.number).columns[0]

print(f"\nColumn Selected for Transformation: {numeric_column}")

Column Selected for Transformation: Moderate Positive Skew


positive_data = data[data[numeric_column] > 0].copy()


positive_data['Log_Transform'] = np.log(positive_data[numeric_column])


positive_data['Reciprocal_Transform'] = 1 / positive_data[numeric_column]


positive_data['Sqrt_Transform'] = np.sqrt(positive_data[numeric_column])


positive_data['Square_Transform'] = np.square(positive_data[numeric_column])


positive_data['BoxCox_Transform'], lambda_value = boxcox(positive_data[numeric_column])

print(f"\nBox-Cox Lambda Value: {lambda_value}")

Box-Cox Lambda Value: 0.35366969646093577


pt = PowerTransformer(method='yeo-johnson')
data['YeoJohnson_Transform'] = pt.fit_transform(data[[numeric_column]])


scaler = StandardScaler()
data['Standard_Scaled'] = scaler.fit_transform(data[[numeric_column]])


positive_data.to_csv('Transformed_Positive_Data.csv', index=False)
data.to_csv('Transformed_Full_Data.csv', index=False)
print("\nTransformation Completed Successfully.")
print("\nTransformed Dataset Preview:")
print(positive_data.head())


Transformation Completed Successfully.

Transformed Dataset Preview:
   Moderate Positive Skew  Highly Positive Skew  Moderate Negative Skew  \
0                0.899990              2.895074               11.180748   
1                1.113554              2.962385               10.842938   
2                1.156830              2.966378               10.817934   
3                1.264131              3.000324               10.764570   
4                1.323914              3.012109               10.753117   

   Highly Negative Skew  Log_Transform  Reciprocal_Transform  Sqrt_Transform  \
0              9.027485      -0.105371              1.111123        0.948678   
1              9.009762       0.107557              0.898026        1.055251   
2              9.006134       0.145684              0.864431        1.075560   
3              9.000125       0.234385              0.791057        1.124336   
4              8.981296       0.280593              0.755336        1.150615   

   Square_Transform  BoxCox_Transform  
0          0.809983         -0.103432  
1          1.240002          0.109628  
2          1.338256          0.149502  
3          1.598027          0.244374  
4          1.752749          0.294988  

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy.stats import boxcox

# Step 2: Load the Dataset
data = pd.read_csv('Data.csv')

print("Original Dataset:")
print(data.head())

# Step 3: Handle Missing Values (Fill numeric columns with mean)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Select a suitable numeric column for transformation
numeric_column = data.select_dtypes(include=np.number).columns[0]

print(f"\nColumn Selected for Transformation: {numeric_column}")

# Keep only positive values for log and boxcox
positive_data = data[data[numeric_column] > 0].copy()

# Step 4: Log Transformation
positive_data['Log_Transform'] = np.log(positive_data[numeric_column])

# Step 5: Reciprocal Transformation
positive_data['Reciprocal_Transform'] = 1 / positive_data[numeric_column]

# Step 6: Square Root Transformation
positive_data['Sqrt_Transform'] = np.sqrt(positive_data[numeric_column])

# Step 7: Square Transformation
positive_data['Square_Transform'] = np.square(positive_data[numeric_column])

# Step 8: Box-Cox Transformation (only positive values)
positive_data['BoxCox_Transform'], lambda_value = boxcox(positive_data[numeric_column])

print(f"\nBox-Cox Lambda Value: {lambda_value}")

# Step 9: Yeo-Johnson Transformation (works with zero/negative values)
pt = PowerTransformer(method='yeo-johnson')
data['YeoJohnson_Transform'] = pt.fit_transform(data[[numeric_column]])

# Standard Scaling
scaler = StandardScaler()
data['Standard_Scaled'] = scaler.fit_transform(data[[numeric_column]])

# Save the transformed dataset
positive_data.to_csv('Transformed_Positive_Data.csv', index=False)
data.to_csv('Transformed_Full_Data.csv', index=False)

print("\nTransformation Completed Successfully.")
print("\nTransformed Dataset Preview:")
print(positive_data.head())


Original Dataset:
   id bin_1 bin_2       City     Ord_1        Ord_2  Target
0   0     F     N      Delhi       Hot  High School       0
1   1     F     Y  Bangalore      Warm      Masters       1
2   2     M     N     Mumbai  Very Hot      Diploma       1
3   3     M     Y    Chennai      Cold    Bachelors       0
4   4     M     Y      Delhi      Cold    Bachelors       1

Column Selected for Transformation: id

Box-Cox Lambda Value: 0.7200338587779628

Transformation Completed Successfully.

Transformed Dataset Preview:
   id bin_1 bin_2       City     Ord_1      Ord_2  Target  Log_Transform  \
1   1     F     Y  Bangalore      Warm    Masters       1       0.000000   
2   2     M     N     Mumbai  Very Hot    Diploma       1       0.693147   
3   3     M     Y    Chennai      Cold  Bachelors       0       1.098612   
4   4     M     Y      Delhi      Cold  Bachelors       1       1.386294   
5   5     F     N      Delhi  Very Hot    Masters       0       1.609438   

   Reciprocal_Transform  Sqrt_Transform  Square_Transform  BoxCox_Transform  
1              1.000000        1.000000                 1          0.000000  
2              0.500000        1.414214                 4          0.898875  
3              0.333333        1.732051                 9          1.674484  
4              0.250000        2.000000                16          2.379521  
5              0.200000        2.236068                25          3.036338  

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy.stats import boxcox

# Step 2: Load the Dataset
data = pd.read_csv('Encoding Data.csv')

print("Original Dataset:")
print(data.head())

# Step 3: Handle Missing Values (Fill numeric columns with mean)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Select a suitable numeric column for transformation
numeric_column = data.select_dtypes(include=np.number).columns[0]

print(f"\nColumn Selected for Transformation: {numeric_column}")

# Keep only positive values for log and boxcox
positive_data = data[data[numeric_column] > 0].copy()

# Step 4: Log Transformation
positive_data['Log_Transform'] = np.log(positive_data[numeric_column])

# Step 5: Reciprocal Transformation
positive_data['Reciprocal_Transform'] = 1 / positive_data[numeric_column]

# Step 6: Square Root Transformation
positive_data['Sqrt_Transform'] = np.sqrt(positive_data[numeric_column])

# Step 7: Square Transformation
positive_data['Square_Transform'] = np.square(positive_data[numeric_column])

# Step 8: Box-Cox Transformation (only positive values)
positive_data['BoxCox_Transform'], lambda_value = boxcox(positive_data[numeric_column])

print(f"\nBox-Cox Lambda Value: {lambda_value}")

# Step 9: Yeo-Johnson Transformation (works with zero/negative values)
pt = PowerTransformer(method='yeo-johnson')
data['YeoJohnson_Transform'] = pt.fit_transform(data[[numeric_column]])

# Standard Scaling
scaler = StandardScaler()
data['Standard_Scaled'] = scaler.fit_transform(data[[numeric_column]])

# Save the transformed dataset
positive_data.to_csv('Transformed_Positive_Data.csv', index=False)
data.to_csv('Transformed_Full_Data.csv', index=False)

print("\nTransformation Completed Successfully.")
print("\nTransformed Dataset Preview:")
print(positive_data.head())


Original Dataset:
   id bin_1 bin_2  nom_0 ord_2
0   0     F     N    Red   Hot
1   1     F     Y   Blue  Warm
2   2     F     N   Blue  Cold
3   3     F     N  Green  Warm
4   4     T     N    Red  Cold

Column Selected for Transformation: id

Box-Cox Lambda Value: 0.7200338587779628

Transformation Completed Successfully.

Transformed Dataset Preview:
   id bin_1 bin_2  nom_0 ord_2  Log_Transform  Reciprocal_Transform  \
1   1     F     Y   Blue  Warm       0.000000              1.000000   
2   2     F     N   Blue  Cold       0.693147              0.500000   
3   3     F     N  Green  Warm       1.098612              0.333333   
4   4     T     N    Red  Cold       1.386294              0.250000   
5   5     T     N  Green   Hot       1.609438              0.200000   

   Sqrt_Transform  Square_Transform  BoxCox_Transform  
1        1.000000                 1          0.000000  
2        1.414214                 4          0.898875  
3        1.732051                 9          1.674484  
4        2.000000                16          2.379521  
5        2.236068                25          3.036338  
# RESULT:
      Thus we performed Feature Encoding and Transformation process and save the data to a file.

       
