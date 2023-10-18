clear all, close all
% size of matrix
k = 10;
L = diag(ones(k,1));
% ls_u = cell((k-1),1);
% ls_l = cell((k-1),1);
% pos = 1;
for idx=k-1:-1:1
    L = L - 1/k * diag(ones(k-idx,1), idx);
    L = L - 1/k * diag(ones(k-idx,1), -idx);
%     l_u = zeros(k,1);
%     l_l = zeros(k,1);
% 
%     l_u(idx+1:end) = ones(k-idx,1);
%     l_l(1:idx) = ones(k-(k-idx),1);
% 
%     ls_u{pos} = l_u;
%     ls_l{pos} = l_l;

%     ls{pos} = -1/k * diag(ones(k,1), idx);
%     ls{pos+1} = -1/k * diag(ones(k-idx,1), -idx);
%     pos = pos +1;
end
(ones(1,k)*L >= 0)
(L*ones(k,1) >= 0)
