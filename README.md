# wine_quality_detection

The project is a multiclass classification problem to classify quality of wine. 
The data includes red wine and white wine. 

red: {3: 10,           white: {3: 20,
      4: 53,                   4: 163,
      5: 681,                  5: 1475,
      6: 638,                  6: 2198,
      7: 199,                  7: 880,
      8: 18}                   8: 175,
                               9: 5}

Left column is label and right column is data scale, the larger label means its quality is greater.
The goal is to model wine quality based on physicochemical tests.

The problem is important because it is a multiclass badly imbalanced problem with the smallest class containing only 10 data and the largest class including 681 data. It has 11 features.
