% Define reference conditions
T_ref = 273.15 + 25; % Reference temperature in Kelvin

% Physical constants
k = 1.38064852e-23;  % Boltzmann constant in [J/(K)]
q = 1.60217657e-19;  % Electron charge in [C]
Vth = k * T_ref / q; % Reference thermal voltage

% Initial parameter guess [IL, Is, Rp, Rs, Ns, n, E, sigma_E_sq, Voc]
Isc = 0.11979;
Is = 2.2e-10;
Rp = 187.5;   % Now the fourth index
Rs = 0.3336;
Ns = 1;
n = 1.5;
E = 1; % Initial guess for E
sigma_E_sq = 1e-6; % Variance of noise
Voc = (n*Vth) * log((Isc/Is) + 1);

% Theoretical I-V data 
V = linspace(0, Voc, 100); % Simulated voltage values

% Calculate current I using the single-diode model equation
I= zeros(size(V));
I = Isc - Is * (exp((V + I * Rs) / (n * Vth)) - 1) - (V + I * Rs) / Rp;

% Plot the theoretical I-V curve
figure;
hold on;
plot(V, I, 'b', 'LineWidth', 2); % I-V curve
yline(Isc, 'r--', 'LineWidth', 2, 'DisplayName', 'Saturation Current I_sc'); % Horizontal line for Is
xline(Voc, 'g--', 'LineWidth', 2, 'DisplayName', 'Open-Circuit Voltage V_{oc}'); % Vertical line for Vth

% Labels and title
xlabel('Voltage (V)');
ylabel('Current (A)');
title('Single-Diode Model Fitting');
grid on;
legend({'I-V Curve', 'Saturation Current I_sc', 'Open-Circuit Voltage V_{oc}'}, 'Location', 'best');
hold off;

% Load experimental data
data = readmatrix('LTspice_V-I_data.csv'); % Assuming two columns: Voltage (V) and Current (I)
V_data = data(:,1);
I_data = data(:,2);

% Define the single-diode model function including Voc
single_diode_model = @(theta, V) ...
    theta(1) - theta(2) * (exp((V + theta(4) * theta(1)) / (theta(5) * theta(6) * Vth * theta(7))) - 1) ...
    - (V + theta(4) * theta(1)) / theta(3);
 
% Add noise to E
epsilon_n = sigma_E_sq * randn(size(V_data)); % i.i.d. normal noise
E_noisy = single_diode_model([Isc, Is, Rp, Rs, Ns, n, E, sigma_E_sq, Voc], V_data) + epsilon_n;

% Initial parameter guess [IL, IS, Rp, Rs, Ns, n, E, sigma_E_sq, Voc]
theta0 = [Isc, Is, Rp, Rs, Ns, n, E, sigma_E_sq, Voc];
%theta0 = theta0 .* (1 + 0.2 * randn(size(theta0))); % Add small random noise

% Optimization options
options = optimoptions('lsqcurvefit', 'Algorithm', 'levenberg-marquardt', ...
    'FunctionTolerance', 1e-8, 'OptimalityTolerance', 1e-8, ...
    'MaxIterations', 1000, 'StepTolerance', 1e-10);

% Perform non-linear least squares fitting
theta_ML = lsqcurvefit(single_diode_model, theta0, V_data, E_noisy, [], [], options);

% Extract estimated parameters
IL_est = theta_ML(1);
Is_est = theta_ML(2);
Rp_est = theta_ML(3);
Rs_est = theta_ML(4);
Ns_est = theta_ML(5);
n_est = theta_ML(6);
E_est = theta_ML(7);
sigma_E_sq_est = theta_ML(8);
Voc_est = theta_ML(9);

% Compute variance of noise (epsilon^2)
residuals = E_noisy - single_diode_model(theta_ML, V_data);
sigma_E_sq = var(residuals);

disp('Estimated Parameters:');
disp(['IL  = ', num2str(IL_est)]);
disp(['IS  = ', num2str(Is_est)]);
disp(['Rp  = ', num2str(Rp_est)]);
disp(['Rs  = ', num2str(Rs_est)]);
disp(['Ns  = ', num2str(Ns_est)]);
disp(['n   = ', num2str(n_est)]);
disp(['E   = ', num2str(E_est)]);
disp(['sigma_E_sq = ', num2str(sigma_E_sq_est)]);
disp(['Voc = ', num2str(Voc_est)]);
disp(['Variance of Noise (epsilon^2) = ', num2str(sigma_E_sq)]);

% Estimating the current  
I_est = single_diode_model(theta_ML, V_data);

% Plot the estimated I-V curve
figure;
hold on;
plot(V_data, I_data, 'r','LineWidth', 2, 'DisplayName', 'Measured Data');
plot(V_data, I_est, 'b--', 'LineWidth', 2,'DisplayName', 'Fitted Model');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend;
title('Single-Diode Model Fitting');
grid on;
hold off;

% Compute theoretical I-V curve
V_sim = linspace(min(V_data), max(V_data), 100);
I_sim = single_diode_model(theta_ML, V_sim);
% Number of bootstrap samples
num_bootstrap = 1000;
bootstrap_estimates = zeros(num_bootstrap, length(theta_ML));

% Bootstrapping process
rng(42); % Set random seed for reproducibility
for b = 1:num_bootstrap
    % Sample with replacement
    idx = randi(length(V_data), length(V_data), 1);
    V_sample = V_data(idx);
    E_sample = E_noisy(idx);

    % Fit model to bootstrap sample
    try
        theta_boot = lsqcurvefit(single_diode_model, theta_ML, V_sample, E_sample, [], [], options);
            if all(isfinite(theta_boot))
                bootstrap_estimates(b, :) = theta_boot;
            else
                disp(['NaN/Inf values encountered at iteration ', num2str(b)]);
            end
    catch ME
        disp(['Optimization failed at iteration ', num2str(b), ': ' ME.message]);
    end

    % Store bootstrapped estimates
    bootstrap_estimates(b, :) = theta_boot;
end

% Compute confidence intervals (95% percentile-based)
CI_lower = prctile(bootstrap_estimates, 2.5);
CI_upper = prctile(bootstrap_estimates, 97.5);

% Display results
disp('Estimated Parameters and 95% Bootstrap Confidence Intervals:');
fprintf('IL       = %.6f, CI = [%.6e, %.6e]\n', theta_ML(1), CI_lower(1), CI_upper(1));
fprintf('Is       = %.6f, CI = [%.6e, %.6e]\n', theta_ML(2), CI_lower(2), CI_upper(2));
fprintf('Rp       = %.6f, CI = [%.6e, %.6e]\n', theta_ML(3), CI_lower(3), CI_upper(3));
fprintf('Rs       = %.6f, CI = [%.6e, %.6e]\n', theta_ML(4), CI_lower(4), CI_upper(4));
fprintf('Ns       = %.6f, CI = [%.6e, %.6e]\n', theta_ML(5), CI_lower(5), CI_upper(5));
fprintf('n        = %.6f, CI = [%.6e, %.6e]\n', theta_ML(6), CI_lower(6), CI_upper(6));
fprintf('E        = %.6f, CI = [%.6e, %.6e]\n', theta_ML(7), CI_lower(7), CI_upper(7));
fprintf('sigma_E  = %.6f, CI = [%.6e, %.6e]\n', theta_ML(8), CI_lower(8), CI_upper(8));
fprintf('Voc      = %.6f, CI = [%.6e, %.6e]\n', theta_ML(9), CI_lower(9), CI_upper(9));

% Plot the fitted curve
figure;
hold on;
plot(V_data, E_noisy, 'ro', 'DisplayName', 'Noisy Data');
plot(V_sim, I_sim, 'b--', 'LineWidth', 2, 'DisplayName', 'Fitted Model');
xlabel('Voltage (V)');
ylabel('Current (A)');
title('Single-Diode Model Fitting with Noisy E');
legend;
grid on;
hold off;
