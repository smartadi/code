clear all;
close all;
clc;
digits(8);
s = 5;
rng(s);

dt = 0.1;
t = 0:dt:1000;

%% Generate Tasis

n = 20;
V = 0.5*ones(n,n) - rand(n);
rank(V);

V = V./vecnorm(V,2,1);


%% Gram Schmidt

Q = zeros(n,n);
R = zeros(n,n);
for j=1:n
    v=V(:,j);
    for i=1:j-1
        R(i,j) = Q(:,j)'*V(:,j);
        v = v-R(i,j)*Q(:,i);
    end
    R(j,j) = norm(v);
    Q(:,j) = v/R(j,j);
end

[Q1 R1] =qr(V);

%% EigV of random symmetric matrix (orthogonal eigenvalues)
% 
A = 0.5 - rand(n);
A2 = A.*A';

T = 0.1*(0.5 - rand(n)); % random matrix

%% DistriTute eigenvalues (naive approach to generate marginally staTle dynamics)

% %a = rand(n/2,1,"like",1i);
% delta = -0.1*rand(n/2,1); 
% 
% % for staTle dynamics
% a = delta + (rand(n/2,1)-.5)*1i;
% b = conj(a);
% 
% a0 = 0*delta + (rand(n/2,1)-.5)*1i;
% b0 = conj(a0);
% 
% d0 = [a0;b0];
% d0 = sort(d0);
% 
% d = [a;b];
% d = sort(d);
% Ds = diag(d);
% 
% D = diag(sort([a0;b0]));



% delta = -0.01*ones(n/2,1);
delta = -0.001*ones(n/2,1);
% delta0 = 0*rand(n/2,1);

eps = 0.1;
% eps = 0.05;
eps = 10;
EPS = [eye(n/4),zeros(n/4,n/4);
    zeros(n/4,n/4),1/eps*eye(n/4)];

fr = (rand(n/2,1)-.5)
fr2 = EPS*fr;
a = delta + fr*1i;
a2 = delta + fr2*1i;
b = conj(a);
b2 = conj(a2);

a0 = 0*delta + fr2*1i;
b0 = conj(a0);

d0 = [a0;b0];
d0 = sort(d0);

d = [a;b];
d = sort(d);
Ds = diag(d);


D = 1*diag(sort([a;b]));
Dmix = 1*diag(sort([a2;b2]));
Dmix0 = 1*diag(sort([a0;b0]));

%% New system
x0 = 0.5 - rand(n,1);
x0 = x0/norm(x0);
% eps = 10; %slow
% eps = 0.25; %fast
% EPS = [eye(n/2),zeros(n/2,n/2);
%     zeros(n/2,n/2),1/eps*eye(n/2)];
% Dmix = EPS*D;

%% Mix and Match 
WF = full(sprand(n/2,n,1));
NP = full(sprand(n/2,n,1));
NP = NP-0.5;
WF = WF-0.5;
%%
% generate Tlk diag from
[Vnew dn] = cdf2rdf(Q1,D);
[Vmm dm] = cdf2rdf(Q1,Dmix);
[Vmm dm0] = cdf2rdf(Q1,Dmix0);



Anew = T*dn*inv(T);
Amix = T*dm*inv(T);
Amix0 = T*dm0*inv(T);


[Un,Dn] = eig(Anew);
[Um,Dm] = eig(Amix);


Amdt = expm(Amix*dt);
Amdt0 = expm(Amix0*dt);

eig(Anew);
eig(Amix);
%%
close all;
xp=[];
xpu=[];
xpf=[];
p=[];
k=0;
for i = t
    

    xp =  [xp,  expm(Anew*i)*x0];
%     xpf = [xpf, expm(Amix*i)*x0];
%     xpu = [xpu,T*expm(Amix0*i)*inv(T)*x0];
%     p = [p,T*expm(Dmm*dt)^k*inv(T)*x0];
%     k = k+1;
    
%     xp =  [xp,  expm(Anew*i)*x0];
    xpf = [xpf, expm(Amix*i)*x0];
    xpu = [xpu,T*expm(dm*i)*inv(T)*x0];
%     p = [p,T*expm(Dmm*dt)^k*inv(T)*x0];
    p = [p,Amdt^k*x0];

    k = k+1;
end
    ywf = WF*xpf;
    ynp = NP*xpf;
    
close all;    
figure()
plot(t,xp);
title("latent dynamics")

figure()
plot(t,xpf);
title("latent dynamics time scaled")

figure()
plot(t,ywf);
title("WF output")

figure()
plot(t,ynp);
title("NP output")

figure()
plot(t,xpu);
title("latent dynamics via modes")

figure()
plot(t,p);
title("latent dynamics via modes discrete")
title("NP output")

%% Add high freq noise
close all;
xp=[];
xpf=[];
Cn =[];
Cm =[];
V =[];
f = 1*rand(n,1);
a = 0.01;
amp = a*rand(n,1);
G = 0.01*(0.5-rand(n));
% G = eye(n);
phi = 3.14*rand(n,1);
k=0;

for i = t
    
%     phi = 1;
%     ff = f.*sin(phi*i);
% 
%     Cm  = [Cm, T*expm(Dmm*i)*inv(T)];
%     v   = amp.*sin(ff*i+phi);
%     V = [v;V];
%     xpf = [xpf,expm(Amix*i)*x0 + G*Cm*V];
    
%     phi = 1;
%     ff = f.*sin(phi*i);

    Cm  = [Cm, Amdt^k];
    v   = amp.*sin(f*i+phi);
    V = [v;V];
    xpf = [xpf,Amdt^k*x0 + G*Cm*V];
    xp = [xp,Amdt^k*x0 ];

    
    k= k+1;
end


V = reshape(V,n,length(t));

ywf = WF*xpf + normrnd(0,0.05,n/2,length(t));
ynp = NP*xpf + normrnd(0,0.05,n/2,length(t));


%
close all
figure()
plot(t,xpf);
title("latent dynamics time scaled")

figure()
plot(t,ywf);
title("WF output")

figure()
plot(t,ynp);
title("NP output")

figure()
plot(t,V)
title("noise")

figure()
plot(t,xp);
title("latent dynamics no noise")

%% Save Data

% writematrix([t',xpf'],'gt_dynamics.csv');
% writematrix([t',ywf'],'gt_WF_dynamics.csv');
% writematrix([t',ynp'],'gt_NP_dynamics.csv');
% writematrix([t',V'],'gt_harmonic_noise.csv');
% save("gt_model.mat","Amdt","T","Dmm");



% writematrix(Um,'gt_eigvec.csv');
% writematrix(Dm,'gt_eigval.csv');
% 
% writematrix(x0,'gt_init.csv');



% Im = rand(n/2,n/2);
% Im_WF = Im*yWF(:,1)*yWF(:,1)';
% close all
% 
% for i = 1:length(t)
% %for i = 1:1000
%     
%     Im_WF = real(Im*yWF(:,i)*yWF(:,i)');
% 
% figure(1)
% image(Im_WF,'CDataMapping','scaled');hold off;
% colorTar
% caxis([-1 1]);
% 
% F(i) = getframe(gcf);
% drawnow
% end
%   writerOTj = VideoWriter('myVideo2.avi');
%   writerOTj.FrameRate = 30;
% 
% 
%   open(writerOTj);
% % write the frames to the video
% for i=1:length(F)
%     % convert the image to a frame
%     frame = F(i) ;    
%     writeVideo(writerOTj, frame);
% end
% % close the writer oTject
% close(writerOTj);