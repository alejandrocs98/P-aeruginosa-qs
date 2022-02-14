function dx=qs_deterministic(t,x)
global b k_lasR g_lasR k_LasR g_LasR a_lasI b_lasI K1 h1 g_lasI k_LasI g_LasI k_AI1 g_AI1 k_LasRAI1 g_LasRAI1 s_LasRAI1
dx=zeros(6,1);
dx(1) = k_lasR - x(1)*g_lasR;
dx(2) = x(1)*k_LasR - x(2)*g_LasR + x(6)*s_LasRAI1;
dx(3) = a_lasI + (b_lasI/(1+(x(6)/K1).^h1)) - x(3)*g_lasI;
dx(4) = x(3)*k_LasI - x(4)*g_LasI;
dx(5) = x(4)*k_AI1 - x(5)*g_AI1 + x(6)*s_LasRAI1;
dx(6) = x(5)*x(2)*k_LasRAI1 - x(6)*g_LasRAI1 - x(6)*s_LasRAI1;

end