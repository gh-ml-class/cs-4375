### Sample Output

```
Opening file Boston.csv.
Reading line 1
heading: rm,medv
new length 506
Closing file Boston.csv.
Number of records: 506

Stats for rm
Sum: 3180.03
Mean: 6.28463
Median: 6.2085
Range: 3.561, 8.78

Stats for medv
Sum: 11401.6
Mean: 22.5328
Median: 21.2
Range: 5, 50

 Covariance = 6.31527

 Correlation = 0.979222
```

### R vs C++

R's built-in functions, combined with its expressive syntax, significantly reduces programmer time compared to manually developing equivalent functionality in C++.

### Mean, Median, and Range

Knowing the mean, median, and range of the dataset you're using enables you to make some general assumptions about the type of data you're working with, without needing to explore every individual observation manually. For example, the mean and median help in estimating what are "normal" samples in the dataset, and the range is an instant indication of how wide a variety of values you will need to handle.

### Covariance and Correlation

Covariance and correlation indicate how closely two attributes are correlated. Since machine learning often involves predicting one variable based on another, it is helpful to get an initial sense of whether it is likely that two variables will have an obvious, strong relationship (thus easier to predict) or whether prediction will be more difficult as the correlation between the variables is less clear.
