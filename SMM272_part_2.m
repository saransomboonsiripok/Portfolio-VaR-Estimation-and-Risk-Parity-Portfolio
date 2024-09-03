close all
clear all
clc
warning('off', 'all')
% ===================================================================
% SMM272 - Risk Analysis Coursework
% ===================================================================
% Part 2.) The risk parity portfolio
% Objective: to construct a Risk Parity Portfolio using the parametric approach
% and evaluate its performance compared to an equally weighted portfolio.

%% Importing data
rng(123)

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

% Combine all return together in a single table
data = INTC(:,{'Date'});
data.INTC = INTC.Log_Return;
data.JPM = JPM.Log_Return;
data.AA = AA.Log_Return;
data.PG = PG.Log_Return;
data.MSFT = MSFT.Log_Return;

% Split data into the first and the second half
sz = size(data,1);
first_half = ceil(sz/2);
data_first = data(1:first_half, : );
data_second = data(first_half + 1:end, :);

%% 1.) Building risk parity portfolio
% to build a risk parity portolio on the first half of data
numobs = size(data_first,1);
numassets = size(data_first,2) - 1;
data_first_return = data_first(:,2:end);
data_first_return = data_first_return{:,:};
sigma = cov(data_first_return);
alpha = 0.99;
z = norminv(1 - alpha, 0, 1);
x0 = ones(numassets,1) / numassets;
options = optimoptions('fmincon', 'Display', 'off');
weight_rp = fmincon(@(x) std(x.*sigma*x/(x'*sigma*x)^0.5), x0, [], [], ones(1, numassets), 1, zeros(numassets,1), ones(numassets,1), [],options);
sg2rp = weight_rp'*sigma*weight_rp;
mvar_rp = -z*sigma*weight_rp/sg2rp^0.5;
cvar_rp = weight_rp.*mvar_rp;
cvar_rp_p = cvar_rp/sum(cvar_rp);

weight_stock = array2table([weight_rp, cvar_rp, cvar_rp_p], "RowNames", stock_name, "VariableNames", {'weight for risk parity portfolio','CvaR','%CVaR'});
disp("- Output 1 - Weight for each stock in risk parity portfolio")
disp(weight_stock)

%% 2.) Calculate return for risk parity portfolio and equal weight portfolio
data_second_return = data_second(:,2:end);
data_second_return = data_second_return{:,:};
rp_return = log(exp(data_second_return) * weight_rp);
eq_return = log(exp(data_second_return) * x0);

disp("- Output 2 - Return series comparison between risk parity portfolio and equally weighted portfolio")
figure;
plot(data_second.Date, exp(cumsum(rp_return)));
hold on;
plot(data_second.Date, exp(cumsum(eq_return)));
hold off;

xlabel('Time');
ylabel('Log-Returns');
title('Return series comparison between risk parity portfolio and equally weighted portfolio');
legend('risk parity portfolio', 'equally weighted portfolio');

%% 3.) Calculate performance measurement of each portfolio (Sharpe index and maximum drawdown)

sharpe_mdd_array = zeros(2,3);
sharpe_rp = sqrt(252) * mean(exp(rp_return)-1) / std(exp(rp_return) -1);
sharpe_eq = sqrt(252) * mean(exp(eq_return)-1) / std(exp(eq_return) -1);

[mdd_rp mdd_pr_rp] = maxdrawdown(cumsum(rp_return),'arithmetic');
[mdd_eq mdd_pr_eq] = maxdrawdown(cumsum(eq_return), 'arithmetic');

sharpe_mdd_array(1,1) = sharpe_rp;
sharpe_mdd_array(1,2) = mdd_rp;
sharpe_mdd_array(1,3) = mdd_pr_rp(2) - mdd_pr_rp(1);
sharpe_mdd_array(2,1) = sharpe_eq;
sharpe_mdd_array(2,2) = mdd_eq;
sharpe_mdd_array(2,3) = mdd_pr_eq(2) - mdd_pr_eq(1);
disp('- Output 3 - Sharpe ratio and maximum drawdown of portfolios')
sharpe_mdd_table = array2table(sharpe_mdd_array, "RowNames",{'Risk parity', 'Equally weighted'}, "VariableNames", {'Sharpe ratio','Maximum Drawdown','Drawdown Period'});
disp(sharpe_mdd_table)
%% 4.) Calculate number of VaR violation at 95% confidence interval

% Calculating VaR according to the first half of the data
expected_ret= mean(data_first_return);
alpha = 0.95;
expected_ret_rp = expected_ret * weight_rp;
expected_ret_eq = expected_ret * x0;
var_rp = weight_rp' * sigma * weight_rp;
var_eq = x0' * sigma * x0;
std_rp = var_rp^0.5;
std_eq = var_eq^0.5;
z = norminv(1 - alpha, 0, 1);
VaR_rp = -(expected_ret_rp + (z * std_rp));
VaR_eq = -(expected_ret_eq + (z * std_eq));

% find violations
violation_rp = sum(rp_return < -VaR_rp);
violation_eq = sum(eq_return < -VaR_eq);

violation_table = array2table([violation_rp violation_eq],"RowNames",{'Number of Violation'},"VariableNames",{'Risk Parity','Equally Weight'});
disp("- Output 4 - Violation of risk parity portfolio return and equally weighted portfolio return")
disp(violation_table)

figure;
plot(rp_return);
hold on;
yline(-VaR_rp, 'r-', 'LineWidth', 2);
xlabel('Time');
ylabel('Returns');
xlim([0 size(rp_return,1)])
title('Returns of risk parity portfolio and VaR');
grid on;
hold off;

figure;
plot(eq_return);
hold on;
yline(-VaR_rp, 'r-', 'LineWidth', 2);
xlabel('Time');
ylabel('Returns');
xlim([0 size(eq_return,1)])
title('Returns of equally weighted portfolio and VaR');
grid on;
hold off;
%% Function used in code
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