
[[Data Dictionary](#dictionary)]
[[Project Description](#project_description)]
[[Project Planning](#project_planning)]
[[Project Acquisition](#project_acquisition)]
[[Project Preparation](#project_preparation)]
[[Project Exploration](#project_exploration)]
[[Project Modeling](#project_modeling)]
[[Project Conclusion](#project_conclusion)]


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
| rgdpe | Expenditure-side real GDP at chain PPPs (in mil. 2017USD) | float64
| rgdpo | Output-side real GDP at chained PPPs (in mil. 2017USD) | float64
| pop | Population (in millions) | float64
| emp | Number of persons engaged (in millions) | float64
| avh | Average annual hours worked by persons engaged | float64
| hc | Human capital index, based on years of schooling and returns to education | float64
#### Current price GDP, capital and TFP
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| ccon | Real consumption of households and government, at current PPPs (in mil. 2017USD) | float64
| cda | Real domestic absorption, (real consumption plus investment), at current PPPs (in mil. 2017USD) | float64
| cgdpe | Expenditure-side real GDP at current PPPs (in mil. 2017USD) | float64
| cgdpo | Output-side real GDP at current PPPs (in mil. 2017USD) | float64
| cn | Capital stock at current PPPs (in mil. 2017USD) | float64
| ck | Capital services levels at current PPPs (USA=1) | float64
| ctfp | TFP level at current PPPs (USA=1) | float64
| cwtfp | Welfare-relevant TFP levels at current PPPs (USA=1) | float64
#### National accounts-based variables
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| rgdpna | Real GDP at constant 2017 national prices (in mil. 2017USD) | float64
| rconna | Real consumption at constant 2017 national prices (in mil. 2017USD) | float64
| rdana | Real domestic absorption at constant 2017 national prices (in mil. 2017USD) | float64
| rnna | Capital stock at constant 2017 national prices (in mil. 2017USD) | float64
| rkna | Capital services at constant 2017 national prices (2017=1) | float64
| rtfpna | TFP at constant national prices (2017=1) | float64
| rwtfpna | Welfare-relevant TFP at constant national prices (2017=1) | float64
| labsh | Share of labour compensation in GDP at current national prices | float64
| irr | Real internal rate of return | float64
| delta| Average depreciation rate of the capital stock | float64
#### Exchange rates and GDP price levels
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| xr | Exchange rate, national currency/USD (market+estimated)| float64
| pl_con | Price level of CCON (PPP/XR), price level of USA GDPo in 2017=1 | float64
| pl_da | Price level of CDA (PPP/XR), price level of USA GDPo in 2017=1 | float64
| pl_gdpo | Price level of CGDo (PPP/XR), price level of USA GDPo in 2017=1| float64
##### Data information variables
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| i_cig | 0/1/2/3/4: relative price data for consumption, investment and government is extrapolated (0), benchmark (1), interpolated (2), ICP PPP timeseries: benchmark or interpolated (3) or  ICP PPP timeseries: extrapolated (4) | object
| i_xm | 0/1/2: relative price data for exports and imports is extrapolated (0), benchmark (1) or interpolated (2) | object
| i_xr| 0/1: the exchange rate is market-based (0) or estimated (1) | object
| i_outlier | 0/1: the observation on pl_gdpe or pl_gdpo is not an outlier (0) or an outlier (1) | object
| i_irr | 0/1/2/3: the observation for irr is not an outlier (0), may be biased due to a low capital share (1), hit the lower bound of 1 percent (2), or is an outlier (3) | object
| capital | share (1), hit the lower bound of 1 percent (2), or is an outlier (3) | object
#### Shares in CGDPo
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| csh_c | Share of household consumption at current PPPs | float64
| csh_i | Share of gross capital formation at current PPPs | float64
| csh_g| Share of government consumption at current PPPs | float64
| csh_x | Share of merchandise exports at current PPPs | float64
| csh_m | Share of merchandise imports at current PPPs | float64
| csh_r | Share of residual trade and GDP statistical discrepancy at current PPPs | float64
#### Price levels, expenditure categories and capital
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| pl_c | Price level of household consumption,  price level of USA GDPo in 2017=1 | float64
| pl_i | Price level of capital formation,  price level of USA GDPo in 2017=1 | float64
| pl_g | Price level of government consumption,  price level of USA GDPo in 2017=1 | float64
| pl_x | Price level of exports,  price level of USA GDPo in 2017=1 | float64
| pl_m | Price level of imports,  price level of USA GDPo in 2017=1 | float64
| pl_n | Price level of the capital stock,  price level of USA GDPo in 2017=1 | float64
| pl_k | Price level of the capital services, price level of USA=1 | float64
#### Suicide rates (per 100,000)
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| both_sexes | Suicide rates of both sexes | float64
| female | Suicide rates of women | float64
| male | Suicide rates of men| float64
***

## Project Description and Goals
<a name="project_description"></a>

- Project Description:
 This repository guides through an analysis of databases from the World Health Organization's (WHO) Global Health Observatory (GHO) and the Groningen Growth and Development Centre's (GGDC) Penn World Table (PWT). The former records the suicide mortality rate (per 100,000 population) and the latter captures relative levels of income, output, input and productivity. The two databases were merged into a single pandas DataFrame, indexed by year. 

- Executive Summary:
The target is the age-standardized suicide rate for both sexes among 15 countries. 
Exploratory Data Analysis was conducted under the assumption that Purchasing Power Parity (PPP) would be the fairest way to compared rates across countries. The GHO and PWT aggregated Pandas DataFrame revealed that, after using Recursive Feature Elimination, the 8 PWT elements which were fed into several regression models, culminated in second-degree Polynomial Regression yielding a Root Mean Square Error of .48 and an explained variance of 99%. Assuming this was done correctly the features ['ctfp', 'cwtfp', 'rwtfpna', 'labsh', 'pl_con', 'pl_gdpo', 'pl_c', 'pl_g'] (see data dictionary) can be used to very accurately estimate a country's suicide rate. This goes to show that although suicide is a very delicate and difficult subject, with many elements that exist far outside the consideration of global economics, the state of a country's monetary affairs can be used to estimate the suicide rates within a country; this is aided by how suicide rates have been relatively stable across time. 
        
***
## Project Planning
## <a name="project_planning"></a>
[[Back to top](#top)]

 **Plan** -> Acquire -> Prepare -> Explore -> Model -> Conclude

- Project Planning:
    - Using my previous research from Scope and Methods in Political Science
        - Collect Data on Suicide Rates and Global Financial Data for several countries
        - Merge datasets while retaining as much information as possible
        - Review Machine Learning methodology as this will be almost entirely continuous data
        - Be sure to attribute the sources appropriately and leave ample markdown and code comments
        - Aid the viewer every step of the way
        - Although plenty of qualitative takeaways are already recorded, focus primarily on the quantitative elements here, objectively viewing suicide rates through monetary lenses.
   - Using my training so far at Codeup
       - Be rigorous and attentive to detail
       - Be as detailed as possible without losing the audience's attention
       - Show useful code, but don't feel the need to show everything, that's what .py is for


### Project Outline:
- It generally goes like this: 
- Acquisiton, in this case through website downloadable csv's
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
- Scaling, Feature-Selection, Modeling
- Conclude with the results.

### Initial Hypotheses
- Countries with higher CGDPe and CGDPo per capita will carry a negative relationship, whereby in most cases these countries will have lower suicide rates per 100,000 than lower-ranking PPP polities. 
- Suicide rates will not differ considerably among the country sample pool
***
## Project Acquisition
<a name="project_acquisition"></a>
[[Back to top](#top)]

 Plan -> **Acquire** -> Prepare -> Explore -> Model -> Conclude

Functions used can be found in wrangle.py. 

### **The easiest way to follow along will be to clone this repository**. 

1. Sites for acquisition
- The GGDC PTW [database](https://www.rug.nl/ggdc/productivity/pwt/?lang=en).
    - *Follow the link, click Excel, import to GoogleSheets, and export as CSV*.  
        - Attribution: Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), "The Next Generation of the Penn World Table" American Economic Review, 105(10), 3150-3182, available for download at www.ggdc.net/pwt
- The WHO GHO [database](https://www.rug.nl/ggdc/productivity/pwt/?lang=en).
    - *Follow the link, download data as csv* 
        - Attribution: World Health Organization. 2019. Global Health Observatory Country Views. 
    
To avoid the rigors of creating this dataframe yourself by merging them, clone my combined dataframe [here](https://github.com/nicholas-dougherty/suicide-studies/blob/main/combined.csv).  

##### The rationale for country selection
- The United States as of 2020 ranks 28th in highest suicide rates in the world. 
- I sought to obtain countries in equal number from Europe, Asia, and the Americas, with half ranking above and the other half ranking below. Originally I hoped to include Hong Kong and Taiwan, but since the former is a semi-autonomous region and the latter is still highly disputed in terms of its international role as a polity, I could not retrieve the information needed from both datasets. So I settled for China. Reliable data for South America is in scarce supply and only Cuba was usable, so the Americas only comprise Canada, United States, and Cuba in this dataframe. 
    - The included countries which rank higher than US in this unfortunate dimension of suicide are: Greenland(1), South Korea(3), Kazakhstan(5), Ukraine(11), Japan(17), France(24), Finland(26)
    - Those than rank lower are: Poland(29), Czech Republic(32), Cuba(33), Germany(33), Canada(43), (India(49), Singapore(51), China(54)
    
2. Use pd.read_csv(). 

***
## Project Preparation
<a name="project_preparation"></a>
[[Back to top](#top)]

 Plan -> Acquire -> **Prepare** -> Explore -> Model -> Conclude

Functions used can be found in wrangle.py. 

My User-Defined Function [describe_data()](https://github.com/nicholas-dougherty/suicide-studies/blob/main/describe.py) runs the gamut of describing the initial dataframe. It revealed that 6% of my data frame had missing values. The UDF [data_prep(df)](https://github.com/nicholas-dougherty/suicide-studies/blob/main/prepare.py) deleted the two columns which had more than half their values missing across columns and rows.       

Given the nature of this information, and the high variability of this PWT's numbers, imputation could not be achieved across multiple columns, and there were only three countries responsible for the nulls. So I dropped them from the dataframe, as mentioned earlier.

```
# Lambda function returns all countries that are not equal to the unwanted three
df = df[df['country'].apply(lambda val: all(val != s for s in ['Belarus', 'Bhutan', 'Guyana']))]
```
The only column still in need of having missing values addressed was the average workhours, which was handled by the mean after using seaborns distplots to observed the impact left by it, vs. mode or median. The only countries missing this value were Ukraine and Kazakhstan

```
# fill nas with mean
df['avh'] = df.avh.fillna(df.avh.mean())
```

There are essentially no outliers in this dataframe, so that was not an issue.
***
## Project Exploration
<a name="project_exploration"></a>
[[Back to top](#top)]

 Plan -> Acquire -> Prepare -> **Explore** -> Model -> Conclude

1. Questions we sought to address within Exploration: 
    - How have suicide rates varied over time?
    - Which features should immediately be dropped from consideration for having low correlation to our target?
    - Which features should be dropped to avoid multicollinearity? 
        - heatmaps helped with the second and third
    - following this, do all features have a linear relationship with suicide-rates?
    
2. Hypothesis Testing
                We rejected the null for each continuous variable. 
    - Is the mean suicide rate in South Korea significantly different from all countries' mean?
        - We failed to reject the null.  
    - Is the mean suicide rate in South Korea significantly different from all countries' mean?
        - We failed to reject the null
***
## Project Modeling
## <a name="project_modeling"></a>
[[Back to top](#top)]

 Plan -> Acquire -> Prepare -> Explore-> **Model** -> Conclude
     
- Implemented a Min-Max Scaler
- Established a baseline with the mean
- Used Recursive Feature Elimination to select eight features 
- Conducted OLS, Polynomial, Tweedie, Lasso+Lars, trying different parameters for each.
Second-Degree Polynomial was wildly successful on test, with a RMSE of .48 and 99% explained variance. 

***
## Project Conclusion and Next Steps:
## <a name="project_conclusion"></a>
[[Back to top](#top)]
Plan -> Acquire -> Prepare -> Explore -> Model -> **Conclude**


This research’s principle aim now is to generate a plan of action that will facilitate effective measurement of the relationships between the economic factors which may overtly lead to the Durkheimian models of isolation and despondency that blend into suicidality.       

RFE in conjunction with the results of machine learning insinuate that Total Factor Productivity (TFP)-level at current Purchasing Power Parities (PPP) __ctfp__, Welfare-relevant TFP levels at current PPPs __cwtfp__, Welfare-relevant TFP at Constant National Prices (CNP) __rwtfpna__, Share of labour compensation in Gross Domestic Product (GDP) at CNP __labsh__, Price level of Real consumption of households and governments(CCON) __pl_con__, Price-level of Output-side real GDP at current PPPs __pl_gdpo__, Price-level of household consumption __pl_c__, and price-level of government consumption __pl_g__ are collectively valuable estimators of suicide rates.
This goes to show that although suicide is a very delicate and difficult subject, with many elements that exist far outside the consideration of global economics, the state of a country's monetary affairs can be used to estimate the suicide rates within a country; this is aided by how suicide rates have been relatively stable across time. 

#### Actionable recommendations: 
WHO publishes suicide rates every two-to-three years. Once the PWT and GHO data concerning the impact of COVID is present in both datasets, aggregate this data as well. 
Because suicide rates have been relatively stable across time, the train, validate, test splits should not be too severely impacted, even with the anticipated changes of the upcoming data deluge. It is perhaps still wisest to index based on the years, so that each country's data is represented in totality. 
To further test the limits of these models, more country's can be included, but certain categories may be lost. However, if these eight features at the very least are fully filled-out, then I have high hopes that a model will perform well. 

Although time-series analysis was not performed in any meaningful way here, the index does suit it, and would be worth further investigation. 
[[Back to top](#top)]
