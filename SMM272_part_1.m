close all
clear all
clc
warning('off', 'all')
% ===================================================================
% SMM272 - Risk Analysis Coursework
% ===================================================================
% Statistical Analysis of a portfolio
% Objective: Analyze an equally weighted portfolio composed of five stocks—Intel (INTC), 
% JPMorgan Chase (JPM), Alcoa (AA), Procter & Gamble (PG), and Microsoft (MSFT)—using 
% historical data from January 1, 2014, to December 31, 2023

%% Importing data
INTC = getMarketDataViaYahoo('INTC', '1-Jan-2014', '1-Jan-2024');
JPM = getMarketDataViaYahoo('JPM', '1-Jan-2014', '1-Jan-2024');
AA = getMarketDataViaYahoo('AA', '1-Jan-2014', '1-Jan-2024');
PG = getMarketDataViaYahoo('PG', '1-Jan-2014', '1-Jan-2024');
MSFT = getMarketDataViaYahoo('MSFT', '1-Jan-2014','1-Jan-2024');

% Filter to get just only Adj Close price
INTC = INTC(:,{'Date','AdjClose'});
JPM = JPM(:,{'Date','AdjClose'});
AA = AA(:,{'Date','AdjClose'});
PG = PG(:,{'Date','AdjClose'});
MSFT = MSFT(:,{'Date','AdjClose'});

% Calculate return of each stock
INTC = cal_return(INTC);
JPM = cal_return(JPM);
AA = cal_return(AA);
PG = cal_return(PG);
MSFT = cal_return(MSFT);
stock_tickers = {INTC, JPM, AA, PG, MSFT};
stock_name = {'INTC', 'JPM', 'AA', 'PG', 'MSFT'};

% Create a table for portfolio log return
portfolio_return = INTC(:,{'Date'});
portfolio_return.Returns = log((0.2 * exp(INTC.Log_Return))+(0.2 * exp(JPM.Log_Return))+ ...
    (0.2 * exp(AA.Log_Return))+(0.2 * exp(PG.Log_Return))+(0.2 * exp(MSFT.Log_Return)));

% Save table for cross-checking all calculations
% writetable(INTC, 'data - INTC.xlsx');
% writetable(JPM, 'data - JPM.xlsx');
% writetable(AA, 'data - AA.xlsx');
% writetable(PG, 'data - PG.xlsx');
% writetable(MSFT, 'data - MSFT.xlsx');
% writetable(portfolio_return, 'data - portfolio_return.xlsx');

%% Part 1) Perform a statistical analysis of the portfolio returns

% 1.) Summary statistics
nobs = length(portfolio_return.Returns);
mean_return = mean(portfolio_return.Returns);
var_return = var(portfolio_return.Returns);
std_return = std(portfolio_return.Returns);
min_return = min(portfolio_return.Returns);
max_return = max(portfolio_return.Returns);
skewness_return = skewness(portfolio_return.Returns);
kurtosis_return = kurtosis(portfolio_return.Returns);
annualized_vol = std_return * sqrt(250);

Des_stat = table(nobs,mean_return,var_return,std_return, ...
    min_return,max_return,skewness_return,kurtosis_return);
Des_stat.Properties.VariableNames = {'N. Obs', 'Mean','Var','St.Dev', 'Min', 'Max', 'Skew', 'Kurt'};
Des_stat.Properties.RowNames = {'Portfolio Return'};
disp("- Output 1 : Table of Descriptive Statistics of Portfolio return")
disp(Des_stat)
figure;
histogram(portfolio_return.Returns)
title("Histogram of portfolio return")
xlabel('log-returns')
disp("- Output 2 : Annualized Volatility (Annualized Standard Deviation)")
disp(annualized_vol)

% 2.) Normality of returns
% Histogram
disp('- Output 3 : Histogram of portfolio return')
plot_1 = figure('Color',[1 1 1]);
subplot(2,2,[1:2]);
nbins = round(nobs ^ 0.5);
histogram(portfolio_return.Returns, nbins, 'Normalization','pdf')
hold on
fplot(@(x) normpdf(x, mean_return, std_return), [min_return max_return]);
xlabel('log-returns')
title('Portfolio returns vs Gaussian Density')
xlim([mean_return - (6 * std_return), mean_return + (6 * std_return)])

subplot(2,2,3)
histogram(portfolio_return.Returns, nbins, 'Normalization','pdf')
hold on
fplot(@(x) normpdf(x, mean_return, std_return), [min_return max_return]) 
xlim([[prctile(portfolio_return.Returns,1) prctile(portfolio_return.Returns,45)]])
xlabel('log-returns')
title('Left Tail')

subplot(2,2,4)
histogram(portfolio_return.Returns, nbins, 'Normalization','pdf')
hold on
fplot(@(x) normpdf(x, mean_return, std_return), [prctile(portfolio_return.Returns,60) prctile(portfolio_return.Returns,99)]) 
xlim([[prctile(portfolio_return.Returns,55) prctile(portfolio_return.Returns,99)]])
xlabel('log-returns')
title('Right Tail')

% QQplot
disp('- Output 4 : QQ Plot')
Z = (portfolio_return.Returns - mean_return) / std_return;
plot_2 = figure('Color',[1 1 1]);
qqplot(Z)
title("QQ Plot of Portfolio Return versus Standard Normal")

% Jarque-Bera test
[h, pvalue, jbstat, critval] = jbtest(Z);
disp("- Output 5 : Result of Jarque-Bera test")
fprintf('Result of test (1 = rejecting the null hypothesis and 0 otherwise): %d\n', h)
fprintf('pvalue: %.5f\n', pvalue)
fprintf('jbstat: %.5f\n', jbstat)
fprintf('critical value: %.5f\n', critval)

% 3.) Cluster of loses and gains
disp('- Output 6 : Plot of clusters of losses')
Extreme = zeros(nobs,1);
%Extreme(find(Z>3))  = Z(find(Z>3));
Extreme(find(Z<-3))  = Z(find(Z<-3));
plot3 = figure('Color',[1 1 1]);
plot(abs(Extreme))
xlabel('Days','interpreter','latex')
ylabel('Losses larger than 3 st. dev.')
title('Clustering of extreme losses')

% 4.) Autocorrelation of data
maxlags = 20;
[acf, lags] = autocorr(portfolio_return.Returns, maxlags);
[acfSq, lags] = autocorr(portfolio_return.Returns.^2, maxlags);
% Return
disp('- Output 7 : Autocorrelation plot of portfolio return')
plot4=figure('Color',[1 1 1]);
autocorr(Z, maxlags)
xlim([0.5, maxlags+0.5])
ylim([min(acf(2:end)) max(acf(2:end))])
title('Autocorrelation of portfolio return')

% Squared Return
disp('- Output 8 : Autocorrelation plot of squared return')
plot5 = figure('Color',[1 1 1]);
autocorr((portfolio_return.Returns).^2, maxlags)
xlim([0.5, maxlags+0.5])
ylim([min(acfSq(2:end)) max(acfSq(2:end))])
xlabel('Lag')
ylabel('Sample autocorrelation')
title('Autocorrelation of squared log-returns')

%% Testing for significance of Autocorrelation
BoxPierce  = sum(acf(2:end).^2)*nobs;
LjungBox = sum(acf(2:end).^2./(nobs-[1:maxlags]'))*nobs*(nobs+2);
BoxPierce2  = sum(acfSq(2:end).^2)*nobs;
LjungBox2 = sum(acfSq(2:end).^2./(nobs-[1:maxlags]'))*nobs*(nobs+2);
CV = icdf('chisquare', 0.95, maxlags);
disp('- Output 9 - Autocorrelation test of returns and squared returns')
if LjungBox>CV
    disp('There is evidence of serial correlation in log-returns')
else
    disp('There is no evidence of serial correlation in log-returns')
end

if LjungBox2  > CV
    disp('There is evidence of serial correlation in squared log-returns')
else
    disp('There is no evidence of serial correlation in squared log-returns')
end
Ljung_box_array = [LjungBox CV; LjungBox2 CV];
Ljung_box_table = array2table(Ljung_box_array, "RowNames",["ret", "ret^2"],"VariableNames",["Ljung-Box stat", "critical value"]);
disp(Ljung_box_table)
autocorr_table = table(lags,acf,acfSq);
autocorr_table.Properties.VariableNames = {'Lags', 'ACF Ret', 'ACF Ret^2'};
disp(autocorr_table)

%% Part 2) 90% and 99% confidence level VaR with different approaches

% A:) Guassian Parametric approach
% 1.) Generate empty array for storing values
Var_gaussian_90 = zeros(1,1);
Var_gaussian_99 = zeros(1,1);
% 2.) Determine the row number of 01 July 2014
first_interval = find(portfolio_return.Date == '30-Jun-2014');
% 3.) Use for loop to iterate over the sample and calculate until the last
% observation
alpha_1 = 0.90;
alpha_2 = 0.99;
j = 1;
for i = first_interval:size(portfolio_return,1) - 1
    data = portfolio_return.Returns(j:i,:);
    Var_gaussian_90(j,1) = VaR_gaussian(data, alpha_1);
    Var_gaussian_99(j,1) = VaR_gaussian(data, alpha_2);
    j = j+1;
end
nan_array = NaN(123,1);
portfolio_return.Var_G_90 = [nan_array; Var_gaussian_90];
portfolio_return.Var_G_99 = [nan_array; Var_gaussian_99];

%B:) Historical Simulation with Bootstraping
% 1.) Generate empty array for storing values and specify number of
% bootstrap sample
Var_hs_90 = zeros(1,1);
Var_hs_99 = zeros(1,1);
Nb = 500;

% 2.) Use for loop to iterate over the sample and calculate until the last
% observation
j = 1;
for i = first_interval:size(portfolio_return,1)-1
    data = portfolio_return.Returns(j:i,:);
    T = length(data);
    for a = 1:Nb
        U = randi(T,T,1);
        Simret = data(U);
        VaRb_90(a) = get_risk_sample(Simret, alpha_1);
        VaRb_99(a) = get_risk_sample(Simret, alpha_2);
    end
    Var_hs_90(j,1) = mean(VaRb_90);
    Var_hs_99(j,1) = mean(VaRb_99);
    j = j+1;
end

portfolio_return.Var_HS_90 = [nan_array; Var_hs_90];
portfolio_return.Var_HS_99 = [nan_array; Var_hs_99];

% C:) Fitting student's T-distribution - maximum likelihood
% 1.) Create empty array
Var_st_90 = zeros(1,1);
Var_st_99 = zeros(1,1);

% 3.) Use for loop to iterate over the sample and calculate until the last
% observation
j = 1;
for i = first_interval:size(portfolio_return,1)-1
    data = portfolio_return.Returns(j:i,:);
    phat = mle(data, 'distribution', 'tlocationscale');
    mu_ml = phat(1);
    sg_ml = phat(2);
    nu_ml = phat(3);
    Var_st_90(j,1) = get_risk_student(mu_ml, sg_ml, nu_ml, alpha_1);
    Var_st_99(j,1) = get_risk_student(mu_ml, sg_ml, nu_ml, alpha_2);
    j = j+1;
    if i == first_interval
        plot_x = figure('Color', [1 1 1]);
        fplot(@(x) pdf('tLocationScale',x, mu_ml, sg_ml, nu_ml),[min(data) max(data)],'b')
        hold on
        fplot(@(x) normpdf(x,mean(data),std(data)),[min(data) max(data)])
        hold on
        histogram(data,round(sqrt(size(data,1))),'normalization','pdf')
        xlim([quantile(data, 0.001) quantile(data, 0.999)])
        legend('Student''s T (MLE)','Gaussian','Empirical', ...
    'location','best', 'interpreter','latex')
        xlabel('log-return', 'interpreter','latex')
        ylabel('pdf', 'interpreter','latex')
        title("Empirical, Guassian and Student's T distribution of the first rolling window")
    end
end

portfolio_return.Var_ST_90 = [nan_array; Var_st_90];
portfolio_return.Var_ST_99 = [nan_array; Var_st_99];

VaR_table = rmmissing(portfolio_return);
disp("- Output 10 - VaR from three different method (save to the directory as 'Output 10 - VaR table.xlsx')")
disp(VaR_table(1:5,:))
writetable(VaR_table,'Output 10 - VaR table.xlsx')

%% plotting var
% Convert the 'Date' column to a datetime format if it's not already
dates = VaR_table.Date;

% Plot the actual returns against the estimated VaRs
figure; % Creates a new figure window
hold on; % Holds the plot for multiple lines
plot(dates, VaR_table.Returns); % Plot actual returns
plot(dates, -1 * VaR_table.Var_G_90, 'LineWidth', 1); % Plot VaR with Gaussian method at 90%
plot(dates, -1 * VaR_table.Var_G_99, 'LineWidth', 1); % Plot VaR with Gaussian method at 99%
plot(dates, -1 * VaR_table.Var_HS_90, 'LineWidth', 1); % Plot VaR with Historical Simulation at 90%
plot(dates, -1 * VaR_table.Var_HS_99, 'LineWidth', 1); % Plot VaR with Historical Simulation at 99%
plot(dates, -1 * VaR_table.Var_ST_90, 'LineWidth', 1); % Plot VaR with Student's T distribution at 90%
plot(dates, -1 * VaR_table.Var_ST_99, 'LineWidth', 1); % Plot VaR with Student's T distribution at 99%
hold off;

% Labeling the axes
xlabel('Date');
ylabel('Returns');

% Adding a title
title('Portfolio Returns and VaR Estimates Over Time');

% Adding a legend
legend('Returns', 'Var\_G\_90', 'Var\_G\_99', 'Var\_HS\_90', 'Var\_HS\_99', 'Var\_ST\_90', 'Var\_ST\_99', 'Location', 'best');

% Improve the formatting of the x-axis dates
datetick('x', 'dd-mmm-yyyy', 'keepticks');
xtickangle(45); % Rotate the x-axis labels for better readability
grid on;
%% Part 3) Compute the number of VaR violations

violation_G_90 = sum(VaR_table.Returns < (VaR_table.Var_G_90).*-1);
violation_G_99 = sum(VaR_table.Returns < (VaR_table.Var_G_99).*-1);
violation_HS_90 = sum(VaR_table.Returns < (VaR_table.Var_HS_90).*-1);
violation_HS_99 = sum(VaR_table.Returns < (VaR_table.Var_HS_99).*-1);
violation_ST_90 = sum(VaR_table.Returns < (VaR_table.Var_ST_90).*-1);
violation_ST_99 = sum(VaR_table.Returns < (VaR_table.Var_ST_99).*-1);

col1 = [violation_G_90; violation_HS_90; violation_ST_90];
col2 = [violation_G_99; violation_HS_99; violation_ST_99];
row_names = ['Gaussian parametric', 'Historical data with Bootstraping', "Student's T - Max likelihood"];
violation_table = table(col1, col2, 'RowNames', row_names, 'VariableNames', {'violations (90% CI)', 'violations (99% CI)'});
disp("- Output 11 - number of violations of each model and each confidence level")
disp(violation_table)

%% Part 4) Kupiec Test

[LR_G_90, Pval_G_90] = get_LRuc(1-alpha_1, violation_G_90, size(VaR_table,1));
[LR_G_99, Pval_G_99] = get_LRuc(1-alpha_2, violation_G_99, size(VaR_table,1));
[LR_HS_90, Pval_HS_90] = get_LRuc(1-alpha_1, violation_HS_90, size(VaR_table,1));
[LR_HS_99, Pval_HS_99] = get_LRuc(1-alpha_2, violation_HS_99, size(VaR_table,1));
[LR_ST_90, Pval_ST_90] = get_LRuc(1-alpha_1, violation_ST_90, size(VaR_table,1));
[LR_ST_99, Pval_ST_99] = get_LRuc(1-alpha_2, violation_ST_99, size(VaR_table,1));

col1 = [violation_G_90; violation_G_99; violation_HS_90; violation_HS_99; violation_ST_90; violation_ST_99];
col2 = [LR_G_90; LR_G_99; LR_HS_90; LR_HS_99; LR_ST_90; LR_ST_99];
col3 = [Pval_G_90; Pval_G_99; Pval_HS_90; Pval_HS_99; Pval_ST_90; Pval_ST_99];
row_names = ['Gaussian (CI = 90%)','Gaussian (CI = 99%)','Historical Simulation (CI = 90%)','Historical Simulation (CI = 99%)',"Student's T (CI = 90%)","Student's T (CI = 99%)"];
LR_table = table(col1,col2,col3,'RowNames',row_names,'VariableNames',{'Violations','Likelihood ratio','P value'});

disp("- Output 12 - Violations, LR and Pvalue from Rupiec test of each model and each confidence interval")
disp(LR_table)

%% Part 5.) visualize the result

% 1.) plot the accumulation of violations
disp('- Output 13 : Plot of the occurrences of violations of each model and each confidence interval')
violation_array = zeros(size(VaR_table,1),6);
for i = 3:8
    data = VaR_table(:,i);
    violation = table2array((VaR_table.Returns < (-1 .* data)) + 0);
    violation_array(:,i-2) = violation;
end

figure;
subplot(2, 3, 1);
plot(cumsum(violation_array(:,1)));
title('Gaussian with 90CI');

subplot(2, 3, 2);
plot(cumsum(violation_array(:,3)));
title('Historical Simulation with 90CI');

subplot(2, 3, 3);
plot(cumsum(violation_array(:,5)));
title("Student't with 90CI");

subplot(2, 3, 4);
plot(cumsum(violation_array(:,2)));
title('Gaussian with 99CI');

subplot(2, 3, 5);
plot(cumsum(violation_array(:,4)));
title('Historical Simulation with 99CI');

subplot(2, 3, 6);
plot(cumsum(violation_array(:,6)));
title("Student't with 99CI");

sgtitle('Cumulative occurence of violations');

% 2.) perform independence test

% - create empty array to store value
ind_array = zeros(6,2);
for i = 3:8
    data = VaR_table(:,i);
    violation = table2array((VaR_table.Returns < (-1 .* data)) + 0);
    n01 = 0;
    n00 = 0;
    n11 = 0;
    n10 = 0;
    for j = 1:length(violation) - 1
        if violation(j) == 0
            if violation(j+1) == 1
                n01 = n01+1;
            else
                n00 = n00+1;
            end
        else
            if violation(j+1) == 1
                n11 = n11+1;
            else
                n10 = n10+1;
            end
        end
    end
    [LR, Pvalue] = get_LRind(n00,n01,n10,n11);
    ind_array(i-2,1) = LR;
    ind_array(i-2,2) = Pvalue;
end
row_names = ['Gaussian (CI = 90%)','Gaussian (CI = 99%)','Historical Simulation (CI = 90%)','Historical Simulation (CI = 99%)',"Student's T (CI = 90%)","Student's T (CI = 99%)"];
col_names = {'Likelihood Ratio', 'PValue'};
ind_table = array2table(ind_array, "RowNames", row_names, "VariableNames", col_names);
disp('- Output 14 - Result of independence test')
disp(ind_table)

%% Part 6) Conditional Coverage and Distributional Test

% 1.) Conditional coverage test - according to Christoffersen's book -
% testing both dependency and number of violation

cc_array = [ind_array(1,1) + LR_G_90;
    ind_array(2,1) + LR_G_99;
    ind_array(3,1) + LR_HS_90;
    ind_array(4,1) + LR_HS_99;
    ind_array(5,1) + LR_ST_90;
    ind_array(6,1) + LR_ST_99;];

for i = 1:6
    cc_array(i,2) = 1 - cdf('chi2', cc_array(i,1), 1);
end

row_names = ['Gaussian (CI = 90%)','Gaussian (CI = 99%)','Historical Simulation (CI = 90%)','Historical Simulation (CI = 99%)',"Student's T (CI = 90%)","Student's T (CI = 99%)"];
col_names = {'Likelihood Ratio', 'PValue'};
cc_table = array2table(cc_array, "RowNames", row_names, "VariableNames", col_names);
disp('- Output 15 - Result of conditional coverage test')
disp(cc_table)

% 2.) distributional test
kuiper_array = zeros(3,2);
mean_std_table = portfolio_return(:,{'Date','Returns'});
mean_array_GU = zeros(1,1);
std_array_GU = zeros(1,1);

% A : Gaussian approach
j = 1;
for i = first_interval:size(portfolio_return,1) - 1
    data = portfolio_return.Returns(j:i,:);
    mean_array_GU(j,1) = mean(data);
    std_array_GU(j,1) = std(data);
    j = j+1;
end
return_array = mean_std_table.Returns;
return_array = return_array(124:end,:);
prob_array_GU = normcdf(return_array, mean_array_GU, std_array_GU);
[K_GU, crit_GU] = kuipertest2(prob_array_GU);
kuiper_array(1,1) = K_GU;
kuiper_array(1,2) = crit_GU;
disp('- Output 16 - Histogram of transform probability - Gaussian approach')
figure;
histogram(prob_array_GU)
title('Histogram of transform probability - Gaussian approach')

% B  : Historical simulation with bootstraping
j = 1;
prob_array_HS = zeros(1,1);
for i = first_interval:size(portfolio_return,1)-1
    data = portfolio_return.Returns(j:i,:);
    ret_tocheck = portfolio_return.Returns(i + 1);
    T = length(data);
    for a = 1:Nb
        U = randi(T,T,1);
        Simret = data(U);
        Sorted_ret = sort(Simret);
        F_emp = (linspace(0,1,size(Sorted_ret,1)))';
        [s_unique, idx] = unique(Sorted_ret);
        Sorted_ret = Sorted_ret(idx);
        F_emp = F_emp(idx);
        if ret_tocheck >= max(Sorted_ret)
            F_sub_array(a) = 1;
        elseif ret_tocheck <= min(Sorted_ret)
            F_sub_array(a) = 0;
        else
        F_sub_array(a) = interp1(Sorted_ret, F_emp, ret_tocheck, 'linear');
        end
    end
    prob_array_HS(j,1) = mean(F_sub_array);
    j = j+1;
end
disp('- Output 17 - Histogram of transform probability - Historical Simulation approach')
figure;
histogram(prob_array_HS)
title('Histogram of transform probability - Historical Simulation approach')
[K_HS, crit_HS] = kuipertest2(prob_array_HS);
kuiper_array(2,1) = K_HS;
kuiper_array(2,2) = crit_HS;

% C: Student's T distribution
mu_array = zeros(1,1);
sg_array = zeros(1,1);
nu_array = zeros(1,1);

j = 1;
for i = first_interval:size(portfolio_return,1)-1
    data = portfolio_return.Returns(j:i,:);
    phat = mle(data, 'distribution', 'tlocationscale');
    mu_ml = phat(1);
    sg_ml = phat(2);
    nu_ml = phat(3);
    mu_array(j,1) = mu_ml;
    sg_array(j,1) = sg_ml;
    nu_array(j,1) = nu_ml;
    j = j+1;
end
prob_array_ST = cdf('tLocationScale',return_array,mu_array,sg_array,nu_array);

disp('- Output 18 - Histogram of transform probability - fitting Student t distribution approach')
figure;
histogram(prob_array_ST)
title('Histogram of transform probability - Fitting Student t distribution approach')
[K_ST, crit_ST] = kuipertest2(prob_array_ST);
kuiper_array(3,1) = K_ST;
kuiper_array(3,2) = crit_ST;


kuiper_table = array2table(kuiper_array, "RowNames", ["Gaussian", "Historical Simulation", "fitting Student's T distribution"], 'VariableNames', {'K','Critical value'});
disp('- Output 19 - Result of Kuiper test')
disp(kuiper_table)

%% Functions used in code

% 1.) Function for calculate Normal return and Log return
function[table_out] = cal_return(table_in)
prices = table_in.AdjClose;
norm_ret = diff(prices) ./ prices(1:end-1);
table_in.Normal_Return = [NaN; norm_ret];
log_ret = diff(log(prices));
table_in.Log_Return = [NaN; log_ret];
table_out = table_in;
table_out = table_out(2:end,:);
end

% 2.) Function for calculating VaR using Guassian Parametric Approach
function[VaR_gau] = VaR_gaussian(data_in, alpha)
mu = mean(data_in);
sigma = std(data_in);
z = norminv(1 - alpha,0,1);
VaR_gau = -((mu) + (z * sigma));
end

% 3.) Function for calculating VaR using Bootstrapping
function[VaR_HS] = get_risk_sample(logret, alpha)
[Nobs Nseries] = size(logret);
NConf = length(alpha);
for i=1:NConf
    for j=1:Nseries
    VaR_HS(i,j) = -quantile(logret(:,j), 1-alpha(i));  
    end
end
end

% 4.) Function to calculate VaR using fitting student's T-distribution
% method
function[VaR_ST] = get_risk_student(mu_t, sg_t, df_t, alpha)
z = tinv(1-alpha, df_t);
VaR_ST = -(mu_t+sg_t*z);
end

% 5.) Function to calculate likelihood ratio and pvalue of Kupiec test
function [LR, Pvalue] = get_LRuc(p, N, n)
phat = N/n;
term1 = -2*(n-N).*log(1-p)-N.*2*log(p);
term2 = -2*(n-N).*log(1-N/n)-N.*2.*log(N/n);
if N == 0
    LR = -2*n*log(1-p);
elseif N == n
    LR = -2*n*log(p);
else
    LR = term1 - term2;
end
Pvalue = 1 - cdf('chi2',LR,1);
end

% 6.) Function for conditional coverage test
function[LR, Pvalue] = get_LRind(n00,n01,n10,n11)
p = (n01+n11) / (n01 + n11 + n00 + n10);
term1 = (n00 + n10) * log(1-p);
term2 = (n01 + n11) * log(p);
LR1 = -2 * (term1 + term2);

pie0 = n01 / (n00 + n01);
pie1 = n11 / (n10 + n11);

term3 = n00 * log(1-pie0);
term4 = n01 * log(pie0);
term5 = n10 * log(1-pie1);
term6 = n11 * log(pie1);
LR2 = -2 * (term3 + term4 + term5 + term6);

LR = LR1 - LR2;
Pvalue = 1 - cdf('chi2', LR,1);
end

% 7.) Function for calculating Kuiper's statistic wtih uniform distribution
function[K, critical_val] = kuipertest(data_in)
a = min(data_in);
b = max(data_in);
f = ecdf(data_in);
disp(f.x)
data = (linspace(0,1,size(data_in,1)+1))';
cdf_uniform = @(x)((x - a) / (b - a));
cdf_vals_uniform = cdf_uniform(data);
disp(size(f))
disp(size(cdf_vals_uniform))
ks_plus = max(x - cdf_vals_uniform);
ks_minus = max(cdf_vals_uniform - x);
K = ks_plus + ks_minus;
critical_val = K*(sqrt(size(data_in,1))+0.155+0.24/sqrt(size(data_in,1)));
disp(f)
end

function[K, critical_val] = kuipertest2(data_in)
a = min(data_in);
b = max(data_in);
data = (linspace(0,1,size(data_in,1)+1))';
cdf_uniform = @(x)((x - a) / (b - a));
cdf_vals_uniform = cdf_uniform(data);
n=length(data_in(:));
m=length(data(:));
res=max(n,m);
y=linspace(0,max(max(data_in),max(data)),res);
f1=zeros([res,1]);f2=zeros([res,1]);
for nn=1:res
    f1(nn)=sum(data_in<=y(nn))/n;% this is the cumulative dist func for s1
    f2(nn)=sum(data<=y(nn))/m;
end
[dplus, dpind]=max([0 ; f1-f2]);
[dminus,dmind]=max([0 ; f2-f1]);
K = dplus+dminus;
critical_val = K*(sqrt(n)+0.155+0.24/sqrt(n));
end