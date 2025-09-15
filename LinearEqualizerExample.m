clc;
close all;

%   Equalize a 16-QAM signal that is impaired by a multipath and
%   Gaussian noise channel.
d = randi([0 15], 1000, 1);
x = qammod(d, 16);


channel = comm.RayleighChannel('PathDelays', [0 1], ...
    'AveragePathGains', [0 -9], ...
    'MaximumDopplerShift', 5e-6);


r = awgn(channel(x), 40, 'measured');


eq = comm.LinearEqualizer;
eq.Constellation = qammod(0:15, 16);
[y,e] = eq(r, x(1:100));
plot(abs(e)); xlabel('Symbols'); ylabel('|e|')
scatterplot(y)





% eqlin = comm.LinearEqualizer( ...
%     Algorithm='LMS', ...
%     NumTaps=10);