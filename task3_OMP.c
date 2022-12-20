#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#define eps 1e-6

#define M0 500
#define N0 1000

int M, N;   // размеры домена
int sM, sN; // индексы элементов домена в сетке
int P, Q;   // размеры сетки процессов
int I, J;   // индексы процесса в сетке процессов

#define off 1

#define A1 -1
#define A2  2
#define B1 -2
#define B2  2

const double h1 = (A2 - A1) / (double) M0;
const double h2 = (B2 - B1) / (double) N0;
const double h1_2 = ((A2-A1) / (double) M0) * ((A2-A1) / (double) M0);
const double h2_2 = ((B2-B1) / (double) N0) * ((B2-B1) / (double) N0);


double k(double x, double y) {
	return 4 + x;
}

double phi(double x, double y) {
    double s_xy = x + y;
    return exp(1 - s_xy*s_xy);
}

double q(double x, double y) {
    double s_xy = x + y;
	return s_xy * s_xy;
}

double F(double x, double y) {
    double s_xy = x + y;
    double s_xy_2 = s_xy*s_xy;
    return exp(1 - s_xy_2) * (s_xy_2 - 8*(x+4)*s_xy_2 + 2*s_xy +4*(x+4));

}


double x_i(int i) {
	return A1 + i*h1;
}

double y_j(int j) {
	return B1 + j*h2;
}


double phi_ij(int i, int j) {
    return phi(x_i(i), y_j(j));
}

double q_ij(int i, int j) {
    return q(x_i(i), y_j(j));
}

double F_ij(int i, int j) {
    return F(x_i(i), y_j(j));
}


double a(int i, int j) {
	return k(x_i(i) - 0.5*h1, y_j(j));
}

double b(int i, int j) {
	return k(x_i(i), y_j(j) - 0.5*h2);
}

double w_x(double** w, int i, int j) {
    int p, q;
    p = i + off - sM;
    q = j + off - sN;
	return (w[p+1][q]-w[p][q]) / h1;
}

double w_y(double** w, int i, int j) {
    int p, q;
    p = i + off - sM;
    q = j + off - sN;
	return (w[p][q+1]-w[p][q]) / h2;
}

double aw_x_l(double **w, int i, int j) {
    return a(i, j) * w_x(w, i-1, j);
}

double bw_y_l(double **w, int i, int j) {
    return b(i, j) * w_y(w, i, j-1);
}

double aw_x_l_x(double **w, int i, int j) {
    return (aw_x_l(w, i+1, j) - aw_x_l(w, i, j)) / h1;
}

double bw_y_l_y(double **w, int i, int j) {
    return (bw_y_l(w, i, j+1) - bw_y_l(w, i, j)) / h2;
}

double laplace(double **w, int i, int j) {
    return aw_x_l_x(w, i, j) + bw_y_l_y(w, i, j);
}

double aw_x_l_m_aw(double **w, int i, int j) {
    int p, q;
    p = i + off - sM;
    q = j + off - sN;
    return (aw_x_l(w,i+1,j) - a(i,j)*w[p][q]/h1) / h1;
}

double aw_m_aw_x_l(double **w, int i, int j) {
    int p, q;
    p = i + off - sM;
    q = j + off - sN;
    return (-a(i+1,j)*w[p][q]/h1 - aw_x_l(w,i,j)) / h1;
}

double bw_y_l_m_bw(double **w, int i, int j) {
    int p, q;
    p = i + off - sM;
    q = j + off - sN;
    return (bw_y_l(w,i,j+1) - b(i,j)*w[p][q]/h2) / h2;
}

double bw_m_bw_y_l(double **w, int i, int j) {
    int p, q;
    p = i + off - sM;
    q = j + off - sN;
    return (-b(i,j+1)*w[p][q]/h2 - bw_y_l(w,i,j)) / h2;
}

double qw(double **w, int i, int j) {
    int p, q;
    p = i + off - sM;
    q = j + off - sN;
    return q_ij(i,j)*w[p][q];
}

double** A_w(double **w, double **r) {
    int i, j, p, q;
    #pragma omp parallel for private(i, j, p, q)
    for (p = off; p <= M+off; p++) {
        i = sM + p - off;
        for (q = off; q <= N+off; q++) {
            j = sN + q - off;
            if      (i == 0 || i == M0)
                r[p][q] = w[p][q];
            else if (j == 0 || j == N0)
                r[p][q] = w[p][q];
            else if (i == 1   && j == 1)
                r[p][q] = -aw_x_l_m_aw(w,i,j) - bw_y_l_m_bw(w,i,j) + qw(w,i,j);
            else if (i == 1   && j == N0-1)
                r[p][q] = -aw_x_l_m_aw(w,i,j) - bw_m_bw_y_l(w,i,j) + qw(w,i,j);
            else if (i == M0-1 && j == 1)
                r[p][q] = -aw_m_aw_x_l(w,i,j) - bw_y_l_m_bw(w,i,j) + qw(w,i,j);
            else if (i == M0-1 && j == N0-1)
                r[p][q] = -aw_m_aw_x_l(w,i,j) - bw_m_bw_y_l(w,i,j) + qw(w,i,j);
            else if (i == 1)
                        r[p][q] = -aw_x_l_m_aw(w,i,j) - bw_y_l_y(w,i,j)    + qw(w,i,j);
            else if (i == M0-1)
                        r[p][q] = -aw_m_aw_x_l(w,i,j) - bw_y_l_y(w,i,j)    + qw(w,i,j);
            else if (j == 1)
                        r[p][q] = -aw_x_l_x(w,i,j)    - bw_y_l_m_bw(w,i,j) + qw(w,i,j);
            else if (j == N0-1)
                        r[p][q] = -aw_x_l_x(w,i,j)    - bw_m_bw_y_l(w,i,j) + qw(w,i,j);
            else
                        r[p][q] = -laplace(w,i,j) + qw(w,i,j);
        }
    }
    return r;
}


void init_w(double **w) {
    int i, j, p, q;
    #pragma omp parallel for private(i, j, p, q)
    for (p = off; p <= M+off; p++) {
        i = sM + p - off;
        for (q = off; q <= N+off; q++) {
            j = sN + q - off;
            if      (i == 0 || i == M0)
                w[p][q] = phi_ij(i, j);
            else if (j == 0 || j == N0)
                w[p][q] = phi_ij(i, j);
            else 
                            w[p][q] = 0;
            }
    }
}

void init_B(double **B) {
    int i, j, p, q;
    #pragma omp parallel for private(i, j, p, q)
    for (p = off; p <= M+off; p++) {
        i = sM + p - off;
        for (q = off; q <= N+off; q++) {
            j = sN + q - off;
            if      ((i == 0) || (i == M0))
                B[p][q] = phi_ij(i,j);
            else if ((j == 0) || (j == N0))
                B[p][q] = phi_ij(i,j);
            else if ((i == 1)    && (j == 1))
                B[p][q] = F_ij(i,j) + a(i,j)   * phi_ij(i-1,j)/h1_2 + b(i,j)   * phi_ij(i,j-1)/h2_2;
            else if ((i == 1)    && (j == N0-1))
                B[p][q] = F_ij(i,j) + a(i,j)   * phi_ij(i-1,j)/h1_2 + b(i,j+1) * phi_ij(i,j+1)/h2_2;
            else if ((i == M0-1) && (j == 1))
                B[p][q] = F_ij(i,j) + a(i+1,j) * phi_ij(i+1,j)/h1_2 + b(i,j)   * phi_ij(i,j-1)/h2_2;
            else if ((i == M0-1) && (j == N0-1))
                B[p][q] = F_ij(i,j) + a(i+1,j) * phi_ij(i+1,j)/h1_2 + b(i,j+1) * phi_ij(i,j+1)/h2_2;
            else if (i == 1)
                        B[p][q] = F_ij(i,j) + a(i,j)   * phi_ij(i-1,j)/h1_2;
            else if (i == M0-1)
                        B[p][q] = F_ij(i,j) + a(i+1,j) * phi_ij(i+1,j)/h1_2;
            else if (j == 1)
                        B[p][q] = F_ij(i,j) + b(i,j)   * phi_ij(i,j-1)/h2_2;
            else if (j == N0-1)
                        B[p][q] = F_ij(i,j) + b(i,j+1) * phi_ij(i,j+1)/h2_2;
            else
                B[p][q] = F_ij(i,j);
        }
    }
}

double init_diff(double **w, double **diff) {
    int i, j, p, q;
    double error = 0;
	for (p = off; p <= M+off; p++) {
        i = sM + p - off;
		for (q = off; q <= N+off; q++) {
            j = sN + q - off;
			diff[p][q] = w[p][q] - phi_ij(i, j);
			error = (fabs(diff[p][q]) > error) ? fabs(diff[p][q]) : error;
		}
    }
    return error;
}

double rho(int i, int j) {
    return ((i == 0 || i == M0) ? 0.5 : 1) * ((j == 0 || j == N0) ? 0.5 : 1);
}

double scalar(double **u, double **v) {
        int i, j, p, q;
        double prod = 0, r = 0;
        #pragma omp parallel for private(i, j, p, q) reduce(+:prod)
        for (p = off; p <= M+off; p++) {
        i = sM + p - off;
                for (q = off; q <= N+off; q++) {
            j = sN + q - off;
                        prod += rho(i, j) * u[p][q] * v[p][q];
        }
    }
    MPI_Allreduce(&prod, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return r * h1 * h2;
}

double** sub(double **a, double **b) {
    int i, j;
	#pragma omp parallel for private(i, j)
        for (i = off; i <= M+off; i++)
                for (j = off; j <= N+off; j++)
                        a[i][j] -= b[i][j];
    return a;
}

double** mult(double t, double **r) {
    int i, j;
	#pragma omp parallel for private(i, j)
        for (i = off; i <= M+off; i++)
                for (j = off; j <= N+off; j++)
                        r[i][j] *= t;
    return r;
}

void init_pqij(int size, int rank) {
    M = M0 + 1;
    N = N0 + 1;
    while (size > 1) {
        if (M == N) {
            if (size >= 4) {
                size = size / 4;
                M = M / 2;
                N = N / 2;
            } else {
                size = size / 2;
                if (M % 2 == 0 || N % 2 != 0)
                    M = M / 2;
                else
                    N = N / 2;
            }
        } else {
            size = size / 2;
            if (M > N)
                M = M / 2;
            else
                N = N / 2;
        }
    }
    P = M0 / M;
    Q = N0 / N;
    I = rank % P;
    J = rank / P;
}

void init_s() {
    int i, j;
    sM = M*I;
    for (i = 0; i < I; i++)
        if ((M0+1) % M >= i + 1)
            sM += 1;
    sN = N*J;
    for (j = 0; j < J; j++)
        if ((N0+1) % N >= j + 1)
            sN += 1;
}

void init_MN() {
    if ((M0+1) % M >= I + 1)
        M += 1;
    M -= 1;
    if ((N0+1) % N >= J + 1)
        N += 1;
    N -= 1;
}

void init(int size, int rank) {
    init_pqij(size, rank);
    init_s();
    init_MN();
}

void sync(int size, int rank, double **w, 
          double *l, int rank_l, double *r, int rank_r, 
          double* t, int rank_t, double *b, int rank_b) {
    int i, j;
    for (i = 1; i <= M+1; i++) {
        b[i-1] = w[i][1];
        t[i-1] = w[i][N+1];
    }
    for (j = 1; j <= N+1; j++) {
        l[j-1] = w[1][j];
        r[j-1] = w[M+1][j];
    }
    MPI_Status status;
    MPI_Sendrecv_replace(l, N+1, MPI_DOUBLE, rank_l, 0, 
                                             rank_l, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Sendrecv_replace(r, N+1, MPI_DOUBLE, rank_r, 0, 
                                             rank_r, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Sendrecv_replace(b, M+1, MPI_DOUBLE, rank_b, 0, 
                                             rank_b, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Sendrecv_replace(t, M+1, MPI_DOUBLE, rank_t, 0, 
                                             rank_t, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    for (i = 1; i <= M+1; i++) {
        w[i][0] = b[i-1];
        w[i][N+2] = t[i-1];
    }
    for (j = 1; j <= N+1; j++) {
        w[0][j] = l[j-1];
        w[M+2][j] = r[j-1];
    }
}

double** allocate() {
    int i;
	double **w = (double **) malloc(sizeof(double*)*(M+1+2*off));
	for (i = 0; i <= M+2*off; i++)
		w[i] = (double *) malloc(sizeof(double)*(N+1+2*off));
	return w;
}

void deallocate(double** w) {
    int i;
	for (i = 0; i <= M+2*off; i++)
		free(w[i]);
	free(w);
}

double* allocate_buf_tb() {
    double *v = (double *) malloc(sizeof(double)*(M+1));
    return v;
}

double* allocate_buf_lr() {
    double *v = (double *) malloc(sizeof(double)*(N+1));
    return v;
}

void out(double **a) {
    int i, j;
    for (i = 1; i <= M+1; i++) {
        for (j = 1; j <= N+1; j++) {
            printf("%lf ", a[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) { 
	int iterations = 0;
    int size, rank, rank_r, rank_l, rank_t, rank_b; 
    double error, begin, end, elapsed, max_error, max_elapsed, prec=1.0;
    double **w, **B, **Ar, **r, **diff;
    double *buf_l, *buf_r, *buf_t, *buf_b;
    MPI_Init(&argc, &argv);
    {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        begin = MPI_Wtime();
        
        init(size, rank);
        rank_l = (I != 0)   ? rank - 1 : MPI_PROC_NULL;
        rank_r = (I != P-1) ? rank + 1 : MPI_PROC_NULL;
        rank_b = (J != 0)   ? rank - P : MPI_PROC_NULL;
        rank_t = (J != Q-1) ? rank + P : MPI_PROC_NULL;
        w    = allocate();
        B    = allocate();
        Ar   = allocate();
        r    = allocate();
        diff = allocate();
        buf_l = allocate_buf_lr();
        buf_r = allocate_buf_lr();
        buf_t = allocate_buf_tb();
        buf_b = allocate_buf_tb();
        

        init_w(w);
        init_B(B);
	    while (prec > eps) {
            iterations += 1;
            sync(size, rank, w, buf_l, rank_l, buf_r, rank_r, buf_t, rank_t, buf_b, rank_b);
    		sub(A_w(w, r), B);                          // r  = A*w-B
            sync(size, rank, r, buf_l, rank_l, buf_r, rank_r, buf_t, rank_t, buf_b, rank_b);
		    A_w(r, Ar);                                 // Ar = A*r
		    sub(w, mult(scalar(Ar,r)/scalar(Ar,Ar),r)); // w' = w - [Ar,r]/[Ar,Ar] * r*/
		    prec = sqrt(scalar(r, r));
	    }
        error = init_diff(w, diff);
        MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        free(buf_l);
        free(buf_r);
        free(buf_t);
        free(buf_b);
        deallocate(w);
        deallocate(B);
        deallocate(Ar);
        deallocate(r);
        deallocate(diff);
        
        end = MPI_Wtime();
        elapsed = end - begin;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Size: %d\nM: %d\nN:%d\nIterations: %d\nTime:  %.16lf\nError: %.16lf\n", 
                   size, M0, N0, iterations, elapsed, error);
    }
    MPI_Finalize();
    return 0;
}
