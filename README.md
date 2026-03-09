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

<img width="750" height="368" alt="Screenshot 2026-03-09 230607" src="https://github.com/user-attachments/assets/91421702-eb3b-4058-8596-b691f9dc77f0" />

data.fillna(data.mean(numeric_only=True), inplace=True)


<img width="802" height="457" alt="Screenshot 2026-03-09 230629" src="https://github.com/user-attachments/assets/8a5459b5-22a6-461b-a091-0299906f33b2" />



numeric_column = data.select_dtypes(include=np.number).columns[0]

print(f"\nColumn Selected for Transformation: {numeric_column}")


<img width="576" height="47" alt="Screenshot 2026-03-09 230646" src="https://github.com/user-attachments/assets/70a7fd0b-d686-4084-aa8b-61e2c69fd227" />



positive_data = data[data[numeric_column] > 0].copy()


positive_data['Log_Transform'] = np.log(positive_data[numeric_column])


positive_data['Reciprocal_Transform'] = 1 / positive_data[numeric_column]


positive_data['Sqrt_Transform'] = np.sqrt(positive_data[numeric_column])


positive_data['Square_Transform'] = np.square(positive_data[numeric_column])


positive_data['BoxCox_Transform'], lambda_value = boxcox(positive_data[numeric_column])

print(f"\nBox-Cox Lambda Value: {lambda_value}")


<img width="430" height="33" alt="Screenshot 2026-03-09 230712" src="https://github.com/user-attachments/assets/73c5d330-891f-428a-b146-729794cbe8a5" />


pt = PowerTransformer(method='yeo-johnson')
data['YeoJohnson_Transform'] = pt.fit_transform(data[[numeric_column]])


scaler = StandardScaler()
data['Standard_Scaled'] = scaler.fit_transform(data[[numeric_column]])


positive_data.to_csv('Transformed_Positive_Data.csv', index=False)
data.to_csv('Transformed_Full_Data.csv', index=False)
print("\nTransformation Completed Successfully.")
print("\nTransformed Dataset Preview:")
print(positive_data.head())


<img width="787" height="568" alt="Screenshot 2026-03-09 230741" src="https://github.com/user-attachments/assets/0f099a42-689a-4904-a411-bd10ba3bfdc0" />

 

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


<img width="788" height="693" alt="Screenshot 2026-03-09 230809" src="https://github.com/user-attachments/assets/a77cad28-dd34-42ef-b116-fa1cb9c53fc9" />


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


<img width="692" height="728" alt="Screenshot 2026-03-09 230836" src="https://github.com/user-attachments/assets/e980a7ff-a786-49e9-bc09-e87fbdfd5dd6" />

# RESULT:
      Thus we performed Feature Encoding and Transformation process and save the data to a file.

       
