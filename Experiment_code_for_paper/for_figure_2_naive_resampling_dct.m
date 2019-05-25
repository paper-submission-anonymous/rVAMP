clear all;

% % load data (riboflavin)
% A0 = csvread("./data/riboflavin/X.csv");
% y0 = csvread("./data/riboflavin/y.csv");
% [M, N] = size(A0);

% load data (random dct)
A0 = csvread("./data/random_DCT/random_DCT_X_alpha01.csv");
y0 = csvread("./data/random_DCT/random_DCT_y_alpha01.csv");
[M, N] = size(A0);

% data normalization
A = zeros(M,N);
for i=1:N
    av = mean(A0(:,i));
    sig = std(A0(:,i));
    A(:,i) = (A0(:,i) - av) / sig;
end

Y = zeros(M, 1);
av = mean(y0);
for i=1:M
    Y(i) = y0(i) - av;
end


% % Other parameters
% lambda = 0.023956316391668966 * M; % l1 regularization (riboflavin)
lambda = 0.019965269054807513 * M; % l1 regularization (random DCT alpha=0.1)
w     = 0.5;        % 1: no penalty randomization, 0.5: recommended in stability selection
p_w   = 0.5;        % 0: no penalty randomization, 0.5: recommended in stability selection
tau   = 0.5;        % 1: standard bootstrap,       0.5: recommended in stability selection


%% Numerical sampling using glmnet
NEXP = 100000;  % num of samples
xV=zeros(N, NEXP);
options = glmnetSet;
options.lambda = lambda / (M * tau);
options.intr = false;
options.standardize = false;
options.thresh = 1.0e-12;
options.maxit = 10^8;
tic;

for nexp=1:NEXP
    nexp
    % set seed
    rng(nexp);

    % randomize penalty coeff. 
    r1 = rand(N, 1);
    w_on = r1<p_w;
    w_off = not(w_on);

    % resample data
    r2 = rand(ceil(M*tau), 1);
    Ibs = ceil(r2 * M);
    Ybs = Y(Ibs);
    Abs = A(Ibs, :);    
    M_tmp = size(Ybs, 1);

    % re-werighting predictors
    Amod = zeros(M_tmp, N);
    Amod(:, w_on) = w * Abs(:, w_on);
    Amod(:, w_off) = Abs(:, w_off);       

    % solve (via Glmnet)
    fit = glmnet(Amod, Ybs, 'gaussian', options);

    % get original weight
    xV(w_on, nexp) = w * fit.beta(w_on);
    xV(w_off, nexp) = fit.beta(w_off);
end
t1=toc;

% % save result
% csvwrite("./data/riboflavin/naive_re-estimation_based_result/matlab_result.csv", xV);
csvwrite("./data/random_DCT/naive_re-estimation_based_result/matlab_result.csv", xV);