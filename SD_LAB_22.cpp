#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <mkl.h>
#include <cmath>
#include <algorithm>

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
std::vector<std::vector<double>> multiply_matrices_blas(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b) {
    size_t n = a.size();
    std::vector<std::vector<double>> c(n, std::vector<double>(n, 0.0));

    // Преобразуем матрицы в формат, подходящий для cblas_dgemm (плоский массив, column-major order)
    std::vector<double> a_flat(n * n);
    std::vector<double> b_flat(n * n);
    std::vector<double> c_flat(n * n);

    // Заполняем плоские массивы в column-major порядке
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            a_flat[j * n + i] = a[i][j];
            b_flat[j * n + i] = b[i][j];
        }
    }

    // Параметры для cblas_dgemm
    const CBLAS_LAYOUT order = CblasColMajor; 
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_TRANSPOSE transb = CblasNoTrans;
    const int m = n;
    const int k = n;  
    const int lda = n;
    const int ldb = n;
    const int ldc = n;
    const double alpha = 1.0;
    const double beta = 0.0;

    // Вызываем cblas_dgemm
    cblas_dgemm(order, transa, transb, m, n, k, alpha, a_flat.data(), lda, b_flat.data(), ldb, beta, c_flat.data(), ldc);

    // Копируем результат обратно в двумерный вектор (column-major)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            c[i][j] = c_flat[j * n + i];
        }
    }

    return c;
}

// 3. Блочное перемножение матриц
std::vector<std::vector<double>> multiply_matrices_blocked(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b, int block_size) {
    int n = a.size();
    std::vector<std::vector<double>> c(n, std::vector<double>(n, 0.0));

    // Предварительное выделение памяти для блоков
    std::vector<double> a_block(block_size * block_size);
    std::vector<double> b_block(block_size * block_size);

    // Блочное умножение с копированием блоков
    for (int i_block = 0; i_block < n; i_block += block_size) {
        for (int k_block = 0; k_block < n; k_block += block_size) {

            // 1. Копирование блока A во временный массив (a_block)
            for (int i = 0; i < std::min(block_size, n - i_block); ++i) {
                for (int k = 0; k < std::min(block_size, n - k_block); ++k) {
                    a_block[i * block_size + k] = a[i_block + i][k_block + k];
                }
            }

            for (int j_block = 0; j_block < n; j_block += block_size) {

                // 2. Копирование блока B во временный массив (b_block)
                for (int k = 0; k < std::min(block_size, n - k_block); ++k) {
                    for (int j = 0; j < std::min(block_size, n - j_block); ++j) {
                        b_block[k * block_size + j] = b[k_block + k][j_block + j];
                    }
                }

                // 3. Умножение блоков
                for (int i = 0; i < std::min(block_size, n - i_block); ++i) {
                    for (int j = 0; j < std::min(block_size, n - j_block); ++j) {
                        double sum = 0.0;
                        for (int k = 0; k < std::min(block_size, n - k_block); ++k) {
                            sum += a_block[i * block_size + k] * b_block[k * block_size + j];
                        }
                        c[i_block + i][j_block + j] += sum;
                    }
                }
            }
        }
    }
    return c;
}

// Функция для проверки равенства матриц
bool matrices_equal(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, double epsilon) {
    int n = a.size();
    if (n != b.size()) {
        std::cerr << "Матрицы имеют разный размер" << std::endl;
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (a[i].size() != b[i].size()) {
            std::cerr << "Матрицы имеют разный размер" << std::endl;
            return false;
        }
        for (int j = 0; j < n; ++j) {
            if (std::abs(a[i][j] - b[i][j]) > epsilon) {
                std::cerr << "Матрицы не равны: a[" << i << "][" << j << "] = " << a[i][j] << ", b[" << i << "][" << j << "] = " << b[i][j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main()
{
    int n = 1024;
    double epsilon = 1e-6;

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
    int block_size = 32; //  Размер блока
    std::cout << "Starting blocked multiplication (block size = " << block_size << ")..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> c3 = multiply_matrices_blocked(a, b, block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_blocked = duration.count() / 1000000.0; // Время в секундах
    double mflops_blocked = complexity / time_blocked / 1e6;
    std::cout << "Blocked multiplication took " << time_blocked << " seconds. MFlops: " << mflops_blocked << std::endl;

    // Проверка "равенства" матриц
    std::cout << "Проверка результатов..." << std::endl;
    bool c1_c2_equal = matrices_equal(c1, c2, epsilon);
    bool c1_c3_equal = matrices_equal(c1, c3, epsilon);
    bool c2_c3_equal = matrices_equal(c2, c3, epsilon);

    std::cout << "c1 и c2 равны: " << (c1_c2_equal ? "YES" : "NO") << std::endl;
    std::cout << "c1 и c3 равны: " << (c1_c3_equal ? "YES" : "NO") << std::endl;
    std::cout << "c2 и c3 равны: " << (c2_c3_equal ? "YES" : "NO") << std::endl;

    // Если хотя бы одна проверка не прошла, выводим сообщение об ошибке
    if (!c1_c2_equal || !c1_c3_equal || !c2_c3_equal) {
        std::cerr << "Внимание: Результаты умножения матриц не совпадают!" << std::endl;
    }

    std::cout << "Выполнил студент группы 090304-РПИа-о24 Шестаченко Иван Александрович";

    return 0;
}
