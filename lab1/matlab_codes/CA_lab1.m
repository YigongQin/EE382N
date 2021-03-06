n=1:1:8192;

figure(1);
z=64;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(n, y);
axis([1, 100 ,0,1]);
legend('cache size=64');
xlabel('Matrix Size');
ylabel('Locality');
title('Locality Vs Matrix Size' );

figure(2);
z=2;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(n), y);
hold on;
z=3;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(n), y);
hold on;
z=4;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(n), y);
hold on;
z=8;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(n), y);
hold on;
z=16;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(n), y);
hold on;
z=32;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(n), y);
hold on;
z=64;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(n), y);
legend('cache size=2','cache size=3','cache size=4','cache size=8','cache size=16','cache size=32','cache size=64');
xlabel('Log of Matrix Size');
ylabel('Locality');
axis([-inf, inf ,-0.1 ,1]);

figure(3);
n=8;
z=1:1:8192;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(z), y);
hold on;
n=16;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(z), y);
hold on;
n=32;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(z), y);
hold on;
n=64;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(z), y);
legend('matrix size=8','matrix size=16','matrix size=32','matrix size=64','Location','SouthEast');
xlabel('Log of Cache Size');
ylabel('Locality');
%axis([z+2, z^2+n ,0.74,0.75]);

figure(4);
z=3:1:8192;
b=floor((z./3).^0.5);
y=1-1./(2.*b);
z=[1,2,z];
y=[0,0,y];
plot(log2(z), y);
xlabel('Log of Cache Size');
ylabel('Locality');
%axis([1, 400, 0 ,1]);

figure(5);
n=8;
z=3:1:8192;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(z), y);
hold on;
n=16;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(z), y);
hold on;
n=32;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(z), y);
hold on;
n=64;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(z), y);
legend('matrix size=8','matrix size=16','matrix size=32','matrix size=64','Location','SouthEast');
xlabel('Log of L2 Cache Size');
ylabel('Locality');

figure(6);
n=32;
z=3:1:8192;
b=floor((z./3).^0.5);
y=1-(3/(4*n)) - (3./(4.*b)).*(b<n);
z=[1,2,z];
y=[0,0,y];
plot(log2(z), y);
xlabel('Log of Cache Size');
ylabel('Locality');
%axis([1, 400, 0 ,1]);

figure(7);
z=3:1:8192;
n=32;
y=0.*(z<3)+(n.^2.*(2*n-1)./(4*n.^3)).*(3<=z & z <n+2)+ (3/4 - 1./(2.*n)).*(n+2<=z & z<n.^2+n+1)+(1 - 3./(4.*n)).*(n.^2+n+1<=z);
plot(log2(z), y);
hold on;


b=floor((z./3).^0.5);
y=(1-1./(2.*b)-1./(4*n)).*(b<n)+(1-(3/(4*n))).*(b>=n);
plot(log2(z), y);
xlabel('Log of L2 Cache Size');
ylabel('Locality');
hold on;

y=1-(3/(4*n)) - (3./(4.*b)).*(b<n);
plot(log2(z), y);
legend('Triply-nested ijk', 'Cache-aware', 'Cache-oblivious','Location', 'SouthEast');
axis([0, inf, 0 ,1]);