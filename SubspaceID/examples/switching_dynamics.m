clear all;
close all;
clc;
% dt = 1/35;
dt = 0.0285;
t = 0:dt:dt*1000;

path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-10-03/1";
upath = '/corr/svdSpatialComponents_ortho.npy';
upath = append(path,upath);
vpath = '/corr/svdTemporalComponents_ortho.npy';
vpath = append(path,vpath);


Uu = readUfromNPY(upath);
Vv = readVfromNPY(vpath);

[TT,dims]= size(Vv);

Vv = Vv./vecnorm(Vv,2,2);

disp('loading WF PCA projections')

data1=Vv';
N=10000;
%%
n = 5;
l = n;
Ts = 500;
data_small = data1(1:l,1:Ts);
nn = 8

%   We will now identify this system from the data y 
%   with the subspace identification algorithm: subid
%   
%   The only extra information we need is the "number of block rows" i
%   in the block Hankel matrices.  This number is easily determined
%   as follows:

%   Say we don't know the order, but think it is maximally equal to 10.
%   
       max_order = 10;
%   
%   As described in the help of subid we can determine "i" as follows:
%   
%       i = 2*(max_order)/(number of outputs)
%       must be an integer
       p = 2*(max_order)/l;

  % nn= 10;

  AUX=[];

 [A,du1,C,du2,K,R,AUX] = subid(data_small,[],p,nn,[],[],1);

 [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');


%%
dd = eig(A)
ds = eig(As)
figure()
plot(dd,'ob');hold on
plot(ds,'or');
legend('unstable','stable')




%%

Q = TT/Ts;
% Q = 5.5;
Vd = zeros(nn,nn,floor(Q));
Ds=[];
D=[];
AA=[];
CC=[];

AAs=[];
CCs=[];
LHS = []
RHS = []
for i = 1:1:Q
    data = data1(1:l,Ts*(i-1)+1 : i*Ts);
    % data = data1(2:l,Ts*(i-1)+1 : i*Ts);

    AUX=[];
    [A,du1,C,du2,K,R,AUX] = subid(data,[],p,nn,[],[],1);

    [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data,[],p,nn,AUX,'sv');

    [Ad,Bd,Cd,Dd,Kd,Ro,AUX,ss,res,Lhs,Rhs] = subid_stable_data(data,[],p,nn,AUX,'sv');


    CC = [CC;C]
    CCs = [CC;Cs]

    AA = [AA;A];
    AAs = [AA;As];

    % if i < 11
    LHS = [LHS;Lhs];
    RHS = [RHS;Rhs];
    % end

    %As = As - Ks*Cs;RHS

    [E,V] = eig(As)
    
    Vd(:,:,i) = V;

    D = [D,eig(A)];
    Ds = [Ds,eig(As)];
eig(A)
end
% %%
% close all;
% figure()
% plot(Ds,'or'); hold on;
% axis([-1 1 -1 1])
% title('stable eigenvalues')
% cDs = Ds/dt;
% cD = D/dt;
% 
% t= 1:1:Q;
% 
% figure()
% plot(t,abs(imag(cDs)),'or'); hold on;
% title('Stable continuous time eigenvalues')
% 
% figure()
% plot(t,abs(imag(cD)),'or'); hold on;
% title('Stable continuous')

%%
close all;
figure()
plot(Ds,'or'); hold on;
axis([-1 1 -1 1])
title('stable eigenvalues')


figure()
plot(D,'or'); hold on;
axis([-1 1 -1 1])
title('unstable eigenvalues')
cDs = (Ds-1)/dt;
cD = D/dt;

t= 1:1:Q;

figure()
plot(t,abs(imag(cDs)),'or'); hold on;
title('Stable continuous time eigenvalues')

figure()
plot(t,abs(imag(cD)),'ob'); hold on;
title('unstable continuous')


figure()
plot(cDs,'or'); hold on;
title('Stable continuous time eigenvalues')

figure()
plot(cD,'ob'); hold on;
title('unstable continuous')
%% Residual

% m = 16; n = 8;
% A = randn(m,n);
% b = randn(m,1);
% bnds = randn(n,2);
% l = min( bnds, [], 2 );
% u = max( bnds, [], 2 );
% 
% x_qp = quadprog( 2*A'*A, -2*A'*b, [], [], [], [], l, u );
% 
% 
% 
% cvx_begin
%     variable x(n)
%     minimize( norm(A*x-b) )
%     subject to
%         l <= x <= u
% cvx_end
%% Change in system
eA=[];
eAs=[];

eC=[];
eCs=[];

t= 1:1:Q-1;

for i = 1:1:Q-1
    eA  = [eA; norm(AA(nn*(i-1)+ 1: nn*i,:) - AA(nn*(i)+ 1: nn*(i+1),:))];
    eAs = [eAs; norm(AAs(nn*(i-1)+ 1: nn*i,:) - AAs(nn*(i)+ 1: nn*(i+1),:))];

    eC  = [eC; norm(CC(l*(i-1)+ 1: l*i,:) - CC(l*(i)+ 1: l*(i+1),:))];
    eCs = [eCs; norm(CCs(l*(i-1)+ 1: l*i,:) - CCs(l*(i)+ 1: l*(i+1),:))];

end

figure()
plot(t,eA); hold on;
plot(t,eAs);
title("Fro norm of change in A")

figure()
plot(t,eC); hold on;
plot(t,eCs);
title("Fro norm of change in C")



%% Switching data

[As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');

[Ad,Bd,Cd,Dd,Kd,Ro,AUX,ss,res,Lhs,Rhs] = subid_stable_data(data_small,[],p,nn,AUX,'sv');


T=101;
cvx_begin
    delta = 0.001
    variable Th((nn+l)*T,nn)
    % II = eye(T)
    % TTh = kron(Th,II);
    for i=1:T
        obj = sum(norm(Th((i-1)*(nn+l)+1:(i)*(nn+l),:)*RHS((i-1)*nn+1:i*nn,:)-LHS((i-1)*(nn+l)+1:i*(nn+l),:)));
    end
    
    minimize( obj )
    subject to
        for i = 1:T-1
            norm(Th((i-1)*(nn+l)+1:(i-1)*(nn+l)+nn) - Th((i)*(nn+l)+1:(i)*(nn+l)+nn)) <= delta
        end
            

cvx_end


%% props of A
D=[];
e=[];
for i=1:T-1
    A = Th((i-1)*(nn+l)+1:(i-1)*(nn+l)+nn,:);

    A2 = Th((i)*(nn+l)+1:(i)*(nn+l)+nn,:);


    [E,V] = eig(A);

    D = [D,eigs(A)];
    e=[e;norm(A-A2)];


end


%%

close all;
figure()
plot(e)
%%
figure()
plot(D,'or'); hold on;
axis([-1 1 -1 1])
title('unstable eigenvalues')
cD = D/dt;

t= 1:1:T-1;

figure()
plot(t,abs(imag(cD)),'ob'); hold on;
title('unstable continuous')

figure()
plot(cD,'ob'); hold on;
title('unstable continuous')