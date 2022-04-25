
[[Data Dictionary](#dictionary)]
[[Project Description](#project_description)]
[[Project Planning](#project_planning)]
[[Project Acquisition](#project_acquisition)]
[[Project Preparation](#project_preparation)]
[[Project Exploration](#project_exploration)]


## Data Dictonary
<a name="dictionary"></a>
[[Back to top](#top)]

| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| countrycode | 3-letter ISO country code | object |  
| country | Country name | object
| currency_unit | Unit of currency per country | object
| year | Years between 2000 & 2019 | datetime64
#### Real GDP, employment and population levels
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| rgdpe | Expenditure-side real GDP at chain PPPs (in mil. 2017USD) | object
| rgdpo | Output-side real GDP at chained PPPs (in mil. 2017USD) | object
| pop | Population (in millions) | object
| emp | Number of persons engaged (in millions) | object
| avh | Average annual hours worked by persons engaged | float64
| hc | Human capital index, based on years of schooling and returns to education | float64
#### Current price GDP, capital and TFP
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| ccon | Real consumption of households and government, at current PPPs (in mil. 2017USD) | float64
| cda | Real domestic absorption, (real consumption plus investment), at current PPPs (in mil. 2017USD) | float64
| cgdpe | Expenditure-side real GDP at current PPPs (in mil. 2017USD) | float64
| cgdpo | Output-side real GDP at current PPPs (in mil. 2017USD) | object
| cn | Capital stock at current PPPs (in mil. 2017USD) | object
| ck | Capital services levels at current PPPs (USA=1) | object
| cftp | TFP level at current PPPs (USA=1) | object
| cwtfp | Welfare-relevant TFP levels at current PPPs (USA=1) | object
#### National accounts-based variables
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| rgdpna | Real GDP at constant 2017 national prices (in mil. 2017USD) | int64
| rconna | Real consumption at constant 2017 national prices (in mil. 2017USD) | object
| rdana | Real domestic absorption at constant 2017 national prices (in mil. 2017USD) | int64
| rnna | Capital stock at constant 2017 national prices (in mil. 2017USD) | float64
| rkna | Capital services at constant 2017 national prices (2017=1) | float64
| rtfpna | TFP at constant national prices (2017=1) | object
| rwtfpna | Welfare-relevant TFP at constant national prices (2017=1) | int64
| labsh | Share of labour compensation in GDP at current national prices | object
| irr | Real internal rate of return | int64
| delta| Average depreciation rate of the capital stock | float64
#### Exchange rates and GDP price levels
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| xr | Sales divided by quantity per order | float64
| pl_con | Brand name extracted from product_name | object
| pl_da | Time between order date and shipping date | int64
| pl_gdpo | Month the order was placed | object
##### Data information variables
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| i_cig | 0/1/2/3/4: relative price data for consumption, investment and government is extrapolated (0), benchmark (1), interpolated (2), ICP PPP timeseries: benchmark or interpolated (3) or  ICP PPP timeseries: extrapolated (4) | int64
| i_xm | 0/1/2: relative price data for exports and imports is extrapolated (0), benchmark (1) or interpolated (2) | float64
| i_xr| 0/1: the exchange rate is market-based (0) or estimated (1) | float64
| i_outlier | 0/1: the observation on pl_gdpe or pl_gdpo is not an outlier (0) or an outlier (1) | object
| i_irr | 0/1/2/3: the observation for irr is not an outlier (0), may be biased due to a low capital share (1), hit the lower bound of 1 percent (2), or is an outlier (3) | int64
| capital | share (1), hit the lower bound of 1 percent (2), or is an outlier (3) | object
#### Shares in CGDPo
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| csh_c | Share of household consumption at current PPPs | int64
| csh_i | Share of gross capital formation at current PPPs | float64
| csh_g| Share of government consumption at current PPPs | float64
| csh_x | Share of merchandise exports at current PPPs | object
| csh_m | Share of merchandise imports at current PPPs | int64
| csh_r | Share of residual trade and GDP statistical discrepancy at current PPPs | float64
#### Price levels, expenditure categories and capital
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| pl_c | Price level of household consumption,  price level of USA GDPo in 2017=1 | float64
| pl_i | Price level of capital formation,  price level of USA GDPo in 2017=1 | object
| pl_g | Price level of government consumption,  price level of USA GDPo in 2017=1 | int64
| pl_x | Price level of exports,  price level of USA GDPo in 2017=1 | float64
| pl_m | Price level of imports,  price level of USA GDPo in 2017=1 | float64
| pl_n | Price level of the capital stock,  price level of USA GDPo in 2017=1 | object
| pl_k | Price level of the capital services, price level of USA=1 | float64
#### Suicide rates (per 100,000)
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| both_sexes | Suicide rates of both sexes | float64
| females | Suicide rates of women | object
| males | Suicide rates of men| float64

## Project Description and Goals
<a name="project_description"></a>

- Project description:
    - Obtain the superstore database as a team, select from the following prompts a mission:
        - 1. CEO: Which customer segment is the best? 
            - If we were going to shift our company to focus more specifically on one customer segment, which one should it be?
        - 2. VP of Marketing: What should we market?
            - We want to launch a new marketing campaign in the near future. How should we target with this campaign? Would you recomend targeting a specific type of customer, product line, or anything else?
        - 3. VP of Sales: What should our sales goals for 2018 be?
            - Are there any additional metrics should we track?
        - 4. VP of Product: Which product line should we expand?
            - Is there a product category that is particularly profitable for us? Does one or another stand out in terms of sales volume? Does this vary by customer segment?
        - Data retrieved from Codeup's MySQL database using a query contained in pd.read_sql.

- Goals:
    - A python script or scripts that automate the data wrangling
    - A notebook or notebooks that include a summary of your data wrangling and exploration
    - A slide deck including an executive summary slide with your recomendations and rationale for them. This should include at least 2 visualizations suitable for presentation.
    - A 5 minute presentation explaining our recomendation


# Project Planning
## <a name="project_planning"></a>
[[Back to top](#top)]

 **Plan** -> Acquire -> Prepare -> Explore 

- Tasking out how we plan to work through the pipeline.
- We have elected to address the Vice President of Product's questions. 

### Target variable
- Profit/profit_per_product

### Initial Focus
- Find distinctions among the Furniture, Office Supplies, and Technology to see how the superstore is profiting from each; noticing trends with this approach will guide our process
- Feature-engineer variables on a product-by-product basis, to show how much profit or sales are made in accordance with the quantity per order
    - Ensure we observe whether or not a discount has been applied before considering a product or subcategory as a boon or blunder
- Of course check immediately for data cleanliness and adjust accordingly.


### Project Outline:
- It generally goes like this: 
- Acquisiton via Codeup Database
- Preparation and pre-preocessing data using Pandas
    - Remove features
        - too many nulls?
        - not helpful to the quest?
    - Create features as needed
    - Handle null values
        - are these fixable or should they just be deleted?
    - Handle outliers
    - Split Data before EDA (not the case for this project)
- Exploratory Data Analysis
     - Visualization using MatPlotLib and Seaborn
- Statistical Testing
- Conclude with the results.

### Hypotheses
- Technology is the most advantageous but underutilized category.
- Technological product lines will be across-the-board worth expanding
- Furniture has been a dangeous and failing pursuit for the superstore and may make its image suffer
- Although most sales happen under office supplies, the profit potential for technology exceeds it. 

# Project Acquisition
<a name="project_acquisition"></a>
[[Back to top](#top)]

 Plan -> **Acquire** -> Prepare -> Explore 

Functions used can be found in wrangle.py. 

1. Acquire the superstore data from the from Codeup's MySQL server, then convert it into a Pandas DataFrame.
```
# If the cached parameter is True, read the csv file on disk in the same folder as this file 
    if os.path.exists('superstore.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('superstore.csv')

    # When there's no cached csv, read the following query from Codeup's MySQL database.
    print('CSV not detected.')
    print('Acquiring data from MySQL database instead.')
    df = pd.read_sql(
        '''
SELECT * FROM orders 
    JOIN customers USING (`Customer ID`)
    JOIN products USING(`Product ID`)
    JOIN categories USING (`Category ID`)
    JOIN regions USING (`Region ID`);             
        '''
                    , get_db_url('superstore_db'))
    
    
    
    print('Acquisition Complete. Dataframe available and is now cached for future use.')
    # create a csv of the dataframe for the sake of efficiency. 
    df.to_csv('superstore.csv', index=False)
```

2. Observe the initial information
    - TAKEAWAYS:
        - There are no nulls and the date-time columns need to be converted to be used
            - No major inconsistencies in the data, but the columns need to be renamed.
```
# Convert column names to snake_case
    df.columns = [col.lower().replace(" ","_").replace("-","_") for col in df.columns]
```

    - There are redundant columns. Removing them is optional. 
3. Used UDF describe data to closely inspect contents.


# Project Preparation
<a name="project_preparation"></a>
[[Back to top](#top)]

 Plan -> Acquire -> **Prepare** -> Explore 

Functions used can be found in wrangle.py. 

1. Clean-up:
    - Take note of the engineered-features
```
# Calculate days between shipment and order placement
    df['days_bw_shipment'] = df['ship_date'] - df['order_date']
# add minutes to the order_date to avoid duplicate values
    df['order_date_anew'] = df['order_date'] + pd.to_timedelta(df.groupby('order_date').cumcount(), unit='h')
# Create product-based columns
    df['profit_per_product'] = df.profit / df.quantity
# add sales per product
    df['sales_per_product'] = df.sales / df.quantity

```
    - Extracting the brand name was tedious, but worked nevertheless. 
    - Initially we split the data before exploration, but later saw this was futile.
    - There was no modeling, and so we ultimately consolidated the test and train sets. 
        - Although retroactively you could see it as us simply ompting not to split. 
        
# Project Exploration
<a name="project_exploration"></a>
[[Back to top](#top)]

 Plan -> Acquire -> Prepare -> **Explore** 

1. Questions we sought to address within Exploration: 
    - Which product line should we expand? 
    - Is there a product category that is particularly profitable for us? 
    - Does one or another stand out in terms of sales volume? 
2. Hypothesis Testing
                We rejected the null for each. 
    - Is Furniture worth keeping under consideration? 
        - No. Furniture profit on average is less than the overall profit avg. 
    - Is Technology a worthwhile business venture? 
        - Absolutely. Technology profit on average is far greater than overall profit avg. 
 CONCLUSION       
3. Answers to the three questions. 
    - Technology under the brands Ativa, Canon, and Konftel
    - Technology invites a lot of room for business growth
    - Office Supplies in their entirety far exceed the other two categories


[[Back to top](#top)]
