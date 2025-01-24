//
// Created by dhonchar on 22 Oct 2024.
//
//#include <Python.h>
//#include <numpy/arrayobject.h>
//#include "src/matmul.h"
#include <immintrin.h>
#include <blis.h>
#include <string.h>      // For memset
#include <nmmintrin.h>
//#include <omp.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

//#include <omp.h>

void matmul_blis_ssyrk(float* A, float* C, int m, int n) {
    float alpha = 1.0;
    float beta = 0.0;
    bli_ssyrk(BLIS_UPPER, BLIS_NO_TRANSPOSE,
              m, n,
              &alpha, A, 1, m,
              &beta, C, 1, m);
}

void matmul_blis_gemm(float* A, float* B, float* C, int m, int n, int k) {
    // Initialize BLIS objects for matrices A, B, and C
    obj_t matA, matB, matC;
    obj_t alpha, beta;
//    float alpha = 1;
// /    float beta = 0;
    // Initialize scalar values for alpha and beta
    bli_obj_scalar_init_detached( BLIS_FLOAT, &alpha );
    bli_obj_scalar_init_detached( BLIS_FLOAT, &beta );

//    // Set alpha to 1.0 and beta to 0.0 (for C = alpha*A*B + beta*C)
    bli_setsc( 1.0, 0.0, &alpha );
    bli_setsc( 0.0, 0.0, &beta );
//
//    // Create BLIS objects from the raw matrix data (A, B, and C)
    bli_obj_create_with_attached_buffer( BLIS_FLOAT, m, k, A, 1, m, &matA );
    bli_obj_create_with_attached_buffer( BLIS_FLOAT, k, n, B, 1, k, &matB );
    bli_obj_create_with_attached_buffer( BLIS_FLOAT, m, n, C, 1, m, &matC );

    // Perform matrix multiplication: C = alpha*A*B + beta*C
    bli_gemm( &alpha, &matA, &matB, &beta, &matC );
    // bli_sgemm
    //  (
    //    BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k,
    //    &alpha,
    //    A, 1, m,
    //    B, 1, k,
    //    &beta,
    //    &matC, 1, m
    //  );
}


// Helper function to sum elements in an AVX register
inline float reduce_add_avx(__m256 vec) {
    __m256 temp = _mm256_hadd_ps(vec, vec);  // Horizontal add
    temp = _mm256_hadd_ps(temp, temp);       // Horizontal add again
    __m128 low = _mm256_castps256_ps128(temp);  // Extract lower half
    __m128 high = _mm256_extractf128_ps(temp, 1);  // Extract upper half
    __m128 sum = _mm_add_ps(low, high);      // Add lower and upper halves
    return _mm_cvtss_f32(sum);               // Return final sum
}



void transpose(float* matrix, float* transposed, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            transposed[j*rows+i] = matrix[i*rows+j];
        }
    }
}

void matmul_naive_binary(char* A, char* B, char* C, const int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < n; p++) {
                C[j * n + i] += (A[p * n + i] & B[j * n + p]);
            }
        }
    }
}


// void float_to_bitpacked(float* A, uint64_t* A_bits, int n) {
//     // For each row
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < n; ++i) {
//         uint64_t bits0 = 0;
//         uint64_t bits1 = 0;
//         int k = 0;
//         // Process first 64 bits
//         for (; k < 64 && k < n; ++k) {
//             if (A[i * n + k] != 0.0f) {
//                 bits0 |= ((uint64_t)1 << k);
//             }
//         }
//         // Process remaining bits
//         for (int idx = 0; k < n; ++k, ++idx) {
//             if (A[i * n + k] != 0.0f) {
//                 bits1 |= ((uint64_t)1 << idx);
//             }
//         }
//         A_bits[i * 2] = bits0;
//         A_bits[i * 2 + 1] = bits1;
//     }
// }
//
// void matmul_bitwise(uint64_t* A_bits, float* C, int n) {
//     // Initialize C to zero
// //    memset(C, 0, n * n * sizeof(float));
//
//     // For each row i
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < n; ++i) {
//         for (int j = i; j < n; ++j) {
//             uint64_t and0 = A_bits[i * 2] & A_bits[j * 2];
//             uint64_t and1 = A_bits[i * 2 + 1] & A_bits[j * 2 + 1];
//
//             // Count the number of set bits
//             float count = (float)(__builtin_popcountll(and0) + __builtin_popcountll(and1));
//
//             // Store the result in C
//             C[i * n + j] = count;
//             if (i != j) {
//                 C[j * n + i] = count; // Exploit symmetry
//             }
//         }
//     }
// }

void float_to_bitpacked(float* A, uint64_t* A_bits, int n) {
    int num_uint64 = (n + 63) / 64;  // Number of uint64_t's per row

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        // For each chunk of 64 bits
        for (int chunk = 0; chunk < num_uint64; ++chunk) {
            uint64_t bits = 0;
            int k_start = chunk * 64;
            int k_end = (k_start + 64 < n) ? k_start + 64 : n;

            for (int k = k_start; k < k_end; ++k) {
                if (A[i * n + k] != 0.0f) {
                    bits |= ((uint64_t)1 << (k - k_start));
                }
            }
            A_bits[i * num_uint64 + chunk] = bits;
        }
    }
}

// void char_to_bitpacked_bool(char* A, uint64_t* A_bits, int n) {
//     int num_uint64 = (n + 63) / 64;  // Number of uint64_t's per row
//
// #pragma omp parallel for schedule(static)
//     for (int i = 0; i < n; ++i) {
//         // For each chunk of 64 bits
//         for (int chunk = 0; chunk < num_uint64; ++chunk) {
//             uint64_t bits = 0;
//             int k_start = chunk * 64;
//             int k_end = (k_start + 64 < n) ? k_start + 64 : n;
//
//             for (int k = k_start; k < k_end; ++k) {
//                 if (A[i * n + k]) {
//                     bits |= ((uint64_t)1 << (k - k_start));
//                 }
//             }
//             A_bits[i * num_uint64 + chunk] = bits;
//         }
//     }
// }

void char_to_bitpacked_bool(char* A, uint64_t* A_bits, int n) {
    int num_uint64 = (n + 63) / 64;  // Number of uint64_t's per row

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        // For each chunk of 64 bits
        for (int chunk = 0; chunk < num_uint64; ++chunk) {
            uint64_t bits = 0;
            int k_start = chunk * 64;
            int k_end = (k_start + 64 <= n) ? k_start + 64 : n;

            int k = k_start;
            // Process 16 bytes at a time using SSE2 instructions
            while (k + 15 < k_end) {
                // Load 16 bytes of data
                __m128i x = _mm_loadu_si128((__m128i*)&A[i * n + k]);

                // Compare each byte with zero to get a mask of the sign bits
                __m128i cmp = _mm_cmpgt_epi8(x, _mm_setzero_si128());

                // Extract the sign bits to create a 16-bit mask
                int mask = _mm_movemask_epi8(cmp);

                // Combine the mask into the bits variable
                bits |= (uint64_t)mask << (k - k_start);

                k += 16;
            }
            // Process any remaining bytes individually
            for (; k < k_end; ++k) {
                if (A[i * n + k]) {
                    bits |= ((uint64_t)1 << (k - k_start));
                }
            }
            A_bits[i * num_uint64 + chunk] = bits;
        }
    }
}

void matmul_bitwise(uint64_t* A_bits, float* C, int n) {
    int num_uint64 = (n + 63) / 64;

    // Optionally initialize C to zero
    // memset(C, 0, n * n * sizeof(float));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float count = 0.0f;
            // Process each chunk
            for (int chunk = 0; chunk < num_uint64; ++chunk) {
                uint64_t and_bits = A_bits[i * num_uint64 + chunk] & A_bits[j * num_uint64 + chunk];
                count += (float)__builtin_popcountll(and_bits);
            }
            C[i * n + j] = count;
            if (i != j) {
                C[j * n + i] = count; // Exploit symmetry
            }
        }
    }
}


void matmul_bitwise_bool(uint64_t* A_bits, float* C, int n) {
    int num_uint64 = (n + 63) / 64;

    // Optionally initialize C to zero
    // memset(C, 0, n * n * sizeof(float));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float count = 0.0f;
            // Process each chunk
            for (int chunk = 0; chunk < num_uint64; ++chunk) {
                uint64_t and_bits = A_bits[i * num_uint64 + chunk] & A_bits[j * num_uint64 + chunk];
                count += (float)__builtin_popcountll(and_bits);
            }
            C[i * n + j] = count;
            if (i != j) {
                C[j * n + i] = count; // Exploit symmetry
            }
        }
    }
}

void float_to_bitpacked_simd(float* A, uint64_t* A_bits, int n) {
    int words_per_row = (n + 63) / 64; // Number of uint64_t words per row

    // Process each row
//#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        uint64_t* row_bits = &A_bits[i * words_per_row];
        memset(row_bits, 0, words_per_row * sizeof(uint64_t));

        int k = 0;
        int bit_idx = 0;
        uint64_t current_word = 0;

        for (; k <= n - 8; k += 8) {
            // Load 8 floats
            __m256 ai = _mm256_loadu_ps(&A[i * n + k]);

            // Compare to zero
            __m256 cmp = _mm256_cmp_ps(ai, _mm256_setzero_ps(), _CMP_NEQ_OQ);

            // Move mask to integer
            int mask = _mm256_movemask_ps(cmp);

            // Set bits accordingly
            current_word |= ((uint64_t)mask) << bit_idx;

            bit_idx += 8;
            if (bit_idx >= 64) {
                // Store the current word
                row_bits[(k - 8) / 64] = current_word;
                current_word = 0;
                bit_idx -= 64;
                if (bit_idx > 0) {
                    current_word = ((uint64_t)mask) >> (8 - bit_idx);
                }
            }
        }

        // Handle remaining elements
        for (; k < n; ++k, ++bit_idx) {
            float val = A[i * n + k];
            if (val != 0.0f) {
                current_word |= ((uint64_t)1 << bit_idx);
            }
            if (bit_idx == 63 || k == n - 1) {
                // Store the current word
                row_bits[k / 64] = current_word;
                current_word = 0;
                bit_idx = -1;
            }
        }
    }
}


// void matmul_optimized_packed(float* A, float* C, int n) {
//     // Allocate memory for bit-packed representation
//     uint64_t* A_bits = (uint64_t*)malloc(n * 2 * sizeof(uint64_t));
//     if (A_bits == NULL) {
//         perror("Memory allocation failed");
//         exit(EXIT_FAILURE);
//     }
//
//     // Convert A to bit-packed representation
//     float_to_bitpacked(A, A_bits, n);
// //    float_to_bitpacked_simd(A, A_bits, n);
//     // Compute C = A * A^T using bitwise operations
//     matmul_bitwise(A_bits, C, n);
//
//     // Free allocated memory
//     free(A_bits);
// }

void matmul_optimized_packed(float* A, float* C, int n) {
    // Calculate the number of uint64_t's needed per row
    int num_uint64 = (n + 63) / 64;

    // Allocate memory for bit-packed representation
    uint64_t* A_bits = (uint64_t*)malloc(n * num_uint64 * sizeof(uint64_t));
    if (A_bits == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Convert A to bit-packed representation
    float_to_bitpacked(A, A_bits, n);

    // Compute C = A * A^T using bitwise operations
    matmul_bitwise(A_bits, C, n);

    // Free allocated memory
    free(A_bits);
}

void matmul_optimized_packed_bool(char* A, float* C, int n) {
    // Calculate the number of uint64_t's needed per row
    int num_uint64 = (n + 63) / 64;

    // Allocate memory for bit-packed representation
    uint64_t* A_bits = (uint64_t*)malloc(n * num_uint64 * sizeof(uint64_t));
    if (A_bits == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Convert A to bit-packed representation
    char_to_bitpacked_bool(A, A_bits, n);

    // Compute C = A * A^T using bitwise operations
    matmul_bitwise(A_bits, C, n);

    // Free allocated memory
    free(A_bits);
}



void matmul_bitwise_simd(uint64_t* A_bits, float* C, int n) {
    int words_per_row = (n + 63) / 64;

    // Initialize C to zero
    memset(C, 0, n * n * sizeof(float));

    // For each row i
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            int total_count = 0;
            int k = 0;

            // Process words in chunks of 4 (256 bits)
            for (; k <= words_per_row - 4; k += 4) {
                // Load 256 bits (4 uint64_t words) from A_bits[i] and A_bits[j]
                __m256i ai = _mm256_loadu_si256((__m256i*)&A_bits[i * words_per_row + k]);
                __m256i aj = _mm256_loadu_si256((__m256i*)&A_bits[j * words_per_row + k]);

                // Bitwise AND
                __m256i and_vec = _mm256_and_si256(ai, aj);

                // Extract individual uint64_t words
                uint64_t val0 = _mm256_extract_epi64(and_vec, 0);
                uint64_t val1 = _mm256_extract_epi64(and_vec, 1);
                uint64_t val2 = _mm256_extract_epi64(and_vec, 2);
                uint64_t val3 = _mm256_extract_epi64(and_vec, 3);

                // Use hardware popcount
                int count = __builtin_popcountll(val0) + __builtin_popcountll(val1) +
                            __builtin_popcountll(val2) + __builtin_popcountll(val3);

                total_count += count;
            }

            // Handle remaining words
            for (; k < words_per_row; ++k) {
                uint64_t ai_word = A_bits[i * words_per_row + k];
                uint64_t aj_word = A_bits[j * words_per_row + k];

                uint64_t and_word = ai_word & aj_word;
                int count = __builtin_popcountll(and_word);
                total_count += count;
            }

            // Store the result in C
            C[i * n + j] = (float)total_count;
            if (i != j) {
                C[j * n + i] = (float)total_count; // Exploit symmetry
            }
        }
    }
}

void matmul_optimized_bitwise_simd(float* A, float* C, int n) {
    int words_per_row = (n + 63) / 64;

    // Allocate memory for bit-packed representation
    uint64_t* A_bits = (uint64_t*)malloc(n * words_per_row * sizeof(uint64_t));
    if (A_bits == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Convert A to bit-packed representation using SIMD
    float_to_bitpacked_simd(A, A_bits, n);

    // Compute C = A * A^T using bitwise operations and SIMD
    matmul_bitwise_simd(A_bits, C, n);

    // Free allocated memory
    free(A_bits);
}

//  /opt/AMD/aocc-compiler-5.0.0/bin/clang -O2 -march=native -mno-avx512f -Lbuild/src -I/usr/local/include/blis/ -L/usr/local/lib/  -shared -o errors.so fast_error.c -lmatmul -lblis-mt
// ldconfig
// /opt/AMD/aocc-compiler-5.0.0/bin/clang -Ofast -march=native -mno-avx512f -Lbuild/src -I/usr/local/include/blis/ -L/usr/local/lib/  -shared -o errors.so fast_error.c -lmatmul -lblis-mt -mfma
