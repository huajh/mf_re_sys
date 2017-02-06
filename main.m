
% matrix factorization for recommender systems
%   @author: Junhao Hua
%   @email:  huajh7@gmail.com
%
%   create time: 2015/2/27
%   last update: 2015/2/27
%
%   reference: Koren, Yehuda, Robert Bell, and Chris Volinsky.
%           "Matrix factorization techniques for recommender systems."
%           Computer 8 (2009): 30-37.
%

load('train_all.mat');

R = L_train;
Y = Rating_train;

[item_num,user_num] = size(R);
lambda = 10;
feat_num = 10;

P = mf_resys_func( Y,R,feat_num,lambda);

fid = fopen('pred_ratings.txt','wt');
 
% for i=1:user_num
%     for j=1:item_num
%         if R(j,i) == 1
%             entry = Y(j,i);
%         else
%             entry = round(P(j,i));
%         end
%         fprintf(fid,'%d %d %d\n',i,j,entry);
%     end
% end
% fclose(fid);

load('data_full.mat');
idx1 = find(L_train==1);
idx2 = find(R==1);
idx = setdiff(idx2,idx1);

% prediction error
pred_mat = zeros(size(P));
pred_mat(idx) = round(P(idx));

pre_err = norm(pred_mat(idx) - Y(idx), 2)^2/ norm(Y(idx),2)^2;
fprintf('\n\nNormalization predication error = %.4f\n', pre_err);






