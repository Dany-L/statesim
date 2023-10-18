s=zpk('s');G=ss(200/(10*s+1)/(0.05*s+1)^2);
systemnames='G';
inputvar='[d;n;r;u]';
outputvar='[G+d-r;u;r-n-d-G]';
input_to_G='[u]';
P=sysic;