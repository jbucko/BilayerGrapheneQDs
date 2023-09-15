CRS_Adjoint.py

-routine to perform mixed global-local optimization for higher states
-number of epochs in gradient descent can be specified
-one should choose the dot (in the code or by flag -d)
we use different notation for the dots: CG2 = QD1, CG3 = QD2, CG9 = QD3
python CRS_Adjoint.py -e 20 -d CG3 (runs for QD2)

-code exits and creates folder with:
    losses_global.csv: contains losses of evaluated points
    UV_global.csv: contains couple of parameters U,V of evaluated points
    results.csv: file containing all important output
