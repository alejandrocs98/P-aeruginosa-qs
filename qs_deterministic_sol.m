global b k_lasR g_lasR k_LasR g_LasR a_lasI b_lasI K1 h1 g_lasI k_LasI g_LasI k_AI1 g_AI1 k_LasRAI1 g_LasRAI1 s_LasRAI1
b = 20;
k_lasR = 200/b;
g_lasR = 0.347;
k_LasR = b*g_LasR;
g_LasR = 0.002;
a_lasI = 0.01;
b_lasI = 130/b;
K1 = 20;
h1 = -2;
g_lasI = 0.347;
k_LasI = b*g_LasI;
g_LasI = 0.01;
k_AI1 = 0.04;
g_AI1 = 0.001;
k_LasRAI1 = 3;
g_LasRAI1 = 0.002;
s_LasRAI1 = 0.1;
[t,x]=ode45(@qs_deterministic,[0 120],[0 0 0 0 0 0]);
plot(t,x)
xlabel('Time')
ylabel('Concentration')
legend('lasR', 'LasR', 'lasI', 'LasI', 'AI1', 'LasRAI1')