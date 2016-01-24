function [ shrink ] = jz_shrink( X,tau )
shrink = sign(X).*max(abs(X)-tau,0);
end

