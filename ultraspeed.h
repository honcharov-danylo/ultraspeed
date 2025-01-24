//
// Created by dhonchar on 23 Oct 2024.
//

void matmul_blis_ssyrk(float* A, float* C, int m, int n);
void matmul_blis_gemm(float* A, float* B, float* C, int m, int n, int k);
void matmul_naive_binary(char* A, char* B, char* C, const int n);

void matmul_optimized_packed(float* A, float* C, int n);
void matmul_optimized_packed_bool(char* A, float* C, int n);
void matmul_optimized_bitwise_simd(float* A, float* C, int n);
