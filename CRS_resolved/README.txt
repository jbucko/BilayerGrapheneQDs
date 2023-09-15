CRS_resolved_subdomain.py

-routine to perform resolved optimization on the subdomain of original domain
-it can be called more times at once to optimize over more subregions at once
-one should choose the dot (in the code or by flag -d)
we use different notation for the dots: CG2 = QD1, CG3 = QD2, CG9 = QD3
-in current setup, domains 50-70 meV for both U,V are split to 5 subdomains along each axis = 25 subdomains labeles by sU1 = {0,1,2,3,4}, sV1 = {0,1,2,3,4}
-to optimize over U = 66-70 meV, V = 50-54 meV, run:  
python CRS_resolved_subdomain.py -sU1 4 -sV1 0 -d CG9 (runs fo QD3)

-code exits and creates folder with:
    losses_global.csv: contains losses of evaluated points
    UV_global.csv: contains couple of parameters U,V of evaluated points
    results.csv: file containing all important output
