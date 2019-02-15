function x_ind = weightedInd(x, x_w, q)

%% This function creates a weighted indicator function of length q 
x_ind = zeros(1,q);
x = round(x);
x_ind(x+1) = x_w;

end