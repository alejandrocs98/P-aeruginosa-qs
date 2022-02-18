function dx=qs_deterministic(t,x)
global b k_lasR g_lasR k_LasR g_LasR a_lasI b_lasI K1 h1 g_lasI k_LasI g_LasI k_AI1 g_AI1 k_LasRAI1 g_LasRAI1 s_LasRAI1 d N d_away g_AI1_ext a_p b_p g_p K2 h2 k_P g_P
dx=zeros(9,1);
dx(1) = k_lasR - x(1)*g_lasR;
dx(2) = x(1)*k_LasR + x(7)*s_LasRAI1 - x(2)*g_LasR;
dx(3) = a_lasI + (b_lasI/(1+(x(7)/K1).^h1)) - x(3)*g_lasI;
dx(4) = x(3)*k_LasI - x(4)*g_LasI;
dx(5) = x(4)*k_AI1 + x(7)*s_LasRAI1 - (d*(x(5)-x(6))) - x(5)*g_AI1;
dx(6) = (N*d*(x(5)-x(6))) - x(6)*(g_AI1_ext + d_away);
dx(7) = x(5)*x(2)*k_LasRAI1 - x(7)*(g_LasRAI1 + s_LasRAI1);
dx(8) = a_p + (b_p/(1+(x(7)/K2).^h2)) - x(8)*g_p;
dx(9) = x(8)*k_P - x(9)*g_P;

end