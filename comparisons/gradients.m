times_symbolic = csvread('times-symbolic.csv', 0)
times_autograd = csvread('times-autograd.csv', 0)

figure;
h1 = histogram(times_symbolic);
hold on
h2 = histogram(times_autograd);

xlabel('Time taken to determine $g(x) = \frac{df}{dx}$ and evaluate g(1.5) \, [s]', 'Interpreter', 'Latex');
ylabel('Frequency');

legend('Symbolic', 'Autodiff');

xlim([0, 3.0e-3])

% h1.Normalization = 'probability';
h1.BinWidth = 0.02e-3;
% h2.Normalization = 'probability';
h2.BinWidth = 0.02e-3;

% xlim([20, 50])

% figure;
% t = linspace(20, 40);

% pd_no_shooting = fitdist(times_without_shooting,'Normal');
% pd_shooting = fitdist(times_with_shooting,'Normal');
% 
% plot(t, pdf(pd_no_shooting, t));
% hold on;
% plot(t, pdf(pd_shooting, t));
% 
% leg1 = legend('Without shooting, m$\mu=29.49 s$', 'With shooting, $\mu=28.75 s$')
% set(leg1,'Interpreter','latex');
% histfit(times_with_shooting);

% mu_shooting = pd_shooting.mu;
% mu_no_shooting = pd_no_shooting.mu;

% percentage_decrease = (mu_no_shooting - mu_shooting)/(mu_no_shooting) * 100;
% xlabel('t [s]')
% ylabel('p(t)')

% [t_modal_no_shooting_frequency, index] = max(h1.BinCounts)
% [t_modal_shooting_frequency, index1] = max(h2.BinCounts)
% 
% t_modal_no_shooting = h1.BinEdges(index)
% t_modal_shooting = h2.BinEdges(index)
% 
% percentage_decrease = 100*(t_modal_no_shooting - t_modal_shooting)/t_modal_no_shooting