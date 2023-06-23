% Controllability check
clc;
close all;

mm=5
B = randn(10,mm);

Co = ctrb(A,B);

r = rank(Co)

if r < n
    disp('Uncontrollable states')
else
    disp('Controllable')



    

%% LQR for 5 dimensional input
close all;
Q = 10000*eye(n);
R = 0.1*eye(mm);

[K,S,e] = dlqr(A,B,Q,R,[]);

T = 1000;
t = 1:1:T;
%amp = randn(n,1);
amp = 0;

w = 0.005;
ref = amp.*sin(w*t).*ones(n,1);

xx = zeros(n,T);
xx(:,1) = x(:,end);
for i=1:T
    xx(:,i+1) = A*xx(:,i) - B*K*(xx(:,i)-ref(:,i));
end


figure()
for i=1:n
    plot(xx(i,:),'Linewidth',2);hold on;
    plot(ref(i,:),'--k')
end
end