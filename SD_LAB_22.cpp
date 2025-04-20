#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>

// Генерация случайной матрицы
std::vector<std::vector<double>> generate_random_matrix(int n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0); // Значения от 0 до 100

    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

// 1. Перемножение матриц по формуле из линейной алгебры (c помощью 3 вложенных циклов for)
std::vector<std::vector<double>> multiply_matrices_ijk(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b)
{
    int n = a.size();
    std::vector<std::vector<double>> c(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

// 2. Перемножение матриц с использованием cblas_dgemm
std::vector<std::vector<double>> multiply_matrices_blas(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b)
{
    int n = a.size();
    std::vector<std::vector<double>> c(n, std::vector<double>(n, 0.0));

    // Преобразуем матрицы в формат, подходящий для cblas_dgemm (плоский массив, column-major order)
    std::vector<double> a_flat(n * n);
    std::vector<double> b_flat(n * n);
    std::vector<double> c_flat(n * n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            a_flat[j * n + i] = a[i][j]; // column-major order
            b_flat[j * n + i] = b[i][j]; // column-major order
        }
    }

    // Параметры для cblas_dgemm
    const enum CBLAS_ORDER order = CblasColMajor;
    const enum CBLAS_TRANSPOSE transa = CblasNoTrans;
    const enum CBLAS_TRANSPOSE transb = CblasNoTrans;
    const int m = n;
    const int lda = n;
    const int ldb = n;
    const int ldc = n;
    const double alpha = 1.0;
    const double beta = 0.0;

    cblas_dgemm(order, transa, transb, m, m, m, alpha, a_flat.data(), lda, b_flat.data(), ldb, beta, c_flat.data(), ldc);

    // Копируем результат обратно в двумерный вектор
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            c[i][j] = c_flat[j * n + i]; // column-major order
        }
    }

    return c;
}

// 3. Блочное перемножение матриц
std::vector<std::vector<double>> multiply_matrices_blocked(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b, int block_size)
{
    int n = a.size();
    std::vector<std::vector<double>> c(n, std::vector<double>(n, 0.0));

    for (int i_block = 0; i_block < n; i_block += block_size)
    {
        for (int j_block = 0; j_block < n; j_block += block_size)
        {
            for (int k_block = 0; k_block < n; k_block += block_size)
            {
                for (int i = i_block; i < std::min(i_block + block_size, n); ++i)
                {
                    for (int j = j_block; j < std::min(j_block + block_size, n); ++j)
                    {
                        for (int k = k_block; k < std::min(k_block + block_size, n); ++k)
                        {
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
    }
    return c;
}

int main()
{
    int n = 1024;

    // Генерируем случайные матрицы
    std::cout << "Generating matrices..." << std::endl;
    std::vector<std::vector<double>> a = generate_random_matrix(n);
    std::vector<std::vector<double>> b = generate_random_matrix(n);

    // Оценка сложности
    double complexity = 2.0 * pow(n, 3);
    std::cout << "Estimated complexity: " << complexity << " Flops" << std::endl;

    // 1. Перемножение матриц по формуле из линейной алгебры
    std::cout << "Starting algebra multiplication..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> c1 = multiply_matrices_ijk(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ijk = duration.count() / 1000000.0; // Время в секундах
    double mflops_ijk = complexity / time_ijk / 1e6;
    std::cout << "algebra multiplication took " << time_ijk << " seconds. MFlops: " << mflops_ijk << std::endl;

    // 2. Перемножение матриц с использованием cblas_dgemm
    std::cout << "Starting BLAS multiplication..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> c2 = multiply_matrices_blas(a, b);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_blas = duration.count() / 1000000.0; // Время в секундах
    double mflops_blas = complexity / time_blas / 1e6;
    std::cout << "BLAS multiplication took " << time_blas << " seconds. MFlops: " << mflops_blas << std::endl;

    // 3. Оптимизированное перемножение матриц (блочное перемножение)
    int block_size = 32; //  Размер блока.  Этот параметр нужно оптимизировать на вашей системе.
    std::cout << "Starting blocked multiplication (block size = " << block_size << ")..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> c3 = multiply_matrices_blocked(a, b, block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_blocked = duration.count() / 1000000.0; // Время в секундах
    double mflops_blocked = complexity / time_blocked / 1e6;
    std::cout << "Blocked multiplication took " << time_blocked << " seconds. MFlops: " << mflops_blocked << std::endl;

    std::cout << "Выполнил студент группы 090304-РПИа-о24 Шестаченко Иван Александрович";

    return 0;
}