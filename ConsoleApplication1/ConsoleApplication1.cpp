#include <omp.h>
#include <iostream>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>

// --------------------- Скалярный алгоритм ------------------------------------------------------------------------------

double rect_integral(double (*func)(const double& x), const double& lower_x, const double& upper_x, const unsigned int& N)
{
	double step = (upper_x - lower_x) / N;
	double x = lower_x + step / 2;  // Идем по серединам отрезков
	double sum = 0;
	while (x < upper_x)
	{
		sum += func(x);  // Поскольку шаг постоянен, умножение на него можно вынести из цикла
		x += step;
	}
	return sum * step;
}

double rect_integral_2d(
	double (*func)(const double& x, const double& y),
	double lower_x,
	const double& upper_x,
	double (*lower_y_func)(const double& x),
	double (*upper_y_func)(const double& x),
	const int& N)
{
	double step_x = (upper_x - lower_x) / N;
	lower_x += step_x / 2;
	double sum = 0;
	for (int i = 0; i < N; ++i)
	{
		double sub_sum = 0;
		double x = lower_x + i * step_x;
		double lower_y = lower_y_func(x);
		double step_y = (upper_y_func(x) - lower_y) / N;
		for (int j = 0; j < N; ++j)
		{
			sub_sum += func(x, lower_y + j * step_y);
		}
		sum += sub_sum * step_y;
	}
	return sum * step_x;
}

double trapz_integral(double (*func)(const double& x), const double& lower_x, const double& upper_x, const unsigned int& N)
{
	double step = (upper_x - lower_x) / N;
	double x = lower_x + step;
	double sum = (func(upper_x) + func(lower_x)) / 2;  //Сразу вычисляем полусумму первого и последнего значений
	while (x < upper_x)  //Идем до предпоследнего элемента
	{
		sum += func(x);
		x += step;
	}
	return sum * step;
}

double trapz_integral_2d(
	double (*func)(const double& x, const double& y),
	double lower_x,
	const double& upper_x,
	double (*lower_y_func)(const double& x),
	double (*upper_y_func)(const double& x),
	const int& N)
{
	double step_x = (upper_x - lower_x) / N;
	double sum = 0;
	for (int i = 1; i < N; ++i)
	{
		double sub_sum = 0;
		double x = lower_x + i * step_x;
		double lower_y = lower_y_func(x);
		double upper_y = upper_y_func(x);
		double step_y = (upper_y - lower_y) / N;
		for (int j = 1; j < N; ++j)
		{
			sub_sum += func(x, lower_y + j * step_y);
		}
		sum += (sub_sum + (func(x, lower_y) + func(x, upper_y)) / 2) * step_y;
	}
	return sum * step_x;
}

// --------------------- Параллельный алгоритм ---------------------------------------------------------------------------

double rect_integral_par(double (*func)(const double& x), double lower_x, const double& upper_x, const int& N, const unsigned int& num_threads)
{
	omp_set_num_threads(num_threads);
	double step = (upper_x - lower_x) / N;
	lower_x += step / 2;
	double sum = 0;
#pragma omp parallel reduction (+: sum)
	{
#pragma omp for
		for (int i = 0; i < N; ++i)
		{
			sum += func(lower_x + i * step);
		}
	} // pragma omp parallel
	return sum * step;
}

double rect_integral_2d_par(
	double (*func)(const double& x, const double& y),
	double lower_x,
	const double& upper_x,
	double (*lower_y_func)(const double& x),
	double (*upper_y_func)(const double& x),
	const int& N,
	const unsigned int& num_threads)
{
	omp_set_num_threads(num_threads);
	double step_x = (upper_x - lower_x) / N;
	lower_x += step_x / 2;
	double sum = 0;
#pragma omp parallel reduction (+: sum) 
	{
#pragma omp for schedule(dynamic)
		for (int i = 0; i < N; ++i)
		{
			double sub_sum = 0;
			double x = lower_x + i * step_x;
			double lower_y = lower_y_func(x);
			double step_y = (upper_y_func(x) - lower_y) / N;
			for (int j = 0; j < N; ++j)
			{
				sub_sum += func(x, lower_y + j * step_y);
			}
			sum += sub_sum * step_y;
		}
	}
	return sum * step_x;
}

double trapz_integral_par(double (*func)(const double& x), double lower_x, const double& upper_x, const int& N, const unsigned int& num_threads)
{
	omp_set_num_threads(num_threads);
	double step = (upper_x - lower_x) / N;
	double sum = 0;
#pragma omp parallel reduction (+: sum)
	{
#pragma omp for
		for (int i = 1; i < N; ++i)
		{
			sum += func(lower_x + i * step);
		}
	} // pragma omp parallel
	sum += (func(upper_x) + func(lower_x)) / 2;
	return sum * step;
}

double trapz_integral_2d_par(
	double (*func)(const double& x, const double& y),
	double lower_x,
	const double& upper_x,
	double (*lower_y_func)(const double& x),
	double (*upper_y_func)(const double& x),
	const int& N,
	const unsigned int& num_threads)
{
	omp_set_num_threads(num_threads);
	double step_x = (upper_x - lower_x) / N;
	double sum = 0;
#pragma omp parallel reduction (+: sum)
	{
#pragma omp for schedule(dynamic)
		for (int i = 1; i < N; ++i)
		{
			double sub_sum = 0;
			double x = lower_x + i * step_x;
			double lower_y = lower_y_func(x);
			double upper_y = upper_y_func(x);
			double step_y = (upper_y - lower_y) / N;
			for (int j = 1; j < N; ++j)
			{
				sub_sum += func(x, lower_y + j * step_y);
			}
			sum += (sub_sum + (func(x, lower_y) + func(x, upper_y)) / 2) * step_y;
		}
	}
	return sum * step_x;
}

//---------------------------------------- Тестовые функции -------------------------------------------
double f(const double& x, const double& y)
{
	return pow(x, 2) + 2 * pow(y, 2);
}

double up_y(const double& x)
{
	return 1 + x;
}

double down_y(const double& x)
{
	return -1;
}


int main()
{
	std::array<std::array<double, 6>, 3> Ints;
	std::array<std::array<double, 6>, 3> T;
	const unsigned int N = 512;
	for (unsigned int i_N = 0; i_N < 6; ++i_N)
	{
		for (unsigned int i_R = 0; i_R < 3; ++i_R)
		{
			std::chrono::duration<double> elapsed_seconds;
			if (i_R == 0)
			{
				auto start{ std::chrono::steady_clock::now() };
				Ints[i_R][i_N] = trapz_integral_2d(f, 0, 1, down_y, up_y, N * (int)pow(2, i_N));
				auto end{ std::chrono::steady_clock::now() };
				elapsed_seconds = end - start;
			}
			else
			{
				auto start{ std::chrono::steady_clock::now() };
				Ints[i_R][i_N] = trapz_integral_2d_par(f, 0, 1, down_y, up_y, N * (int)pow(2, i_N), 2 * i_R);
				auto end{ std::chrono::steady_clock::now() };
				elapsed_seconds = end - start;
			}
			T[i_R][i_N] = elapsed_seconds.count();
		}
	}

	std::ofstream csv("trapz.csv", std::ofstream::out);
	std::cout << "N";
	csv << "N" << ";";
	for (unsigned int i_N = 0; i_N < 6; ++i_N)
	{
		int N = 512 * (int)pow(2, i_N);
		std::cout << "\t | \t\t" << N;
		csv << N << ";";
	}

	for (unsigned int i_R = 0; i_R < 3; ++i_R)
	{
		unsigned int R = (unsigned int)pow(2, i_R);
		std::cout << "\n\nR = " << R << "\nInt ";
		csv << "\nR; " << R << "\nInt;";

		for (unsigned int i_N = 0; i_N < 6; ++i_N)
		{
			std::cout << std::setprecision(5) << std::scientific << "\t | \t" << Ints[i_R][i_N];
			csv << Ints[i_R][i_N] << ";";
		}

		std::cout << "\nT" << i_R + 1;
		csv << "\nT" << i_R + 1 << ';';
		for (unsigned int i_N = 0; i_N < 6; ++i_N)
		{
			std::cout << std::setprecision(5) << std::scientific << "\t | \t" << T[i_R][i_N];
			csv << T[i_R][i_N] << ";";
		}

		if (i_R != 0)
		{
			std::cout << "\nT1 / T" << i_R + 1;
			csv << "\nT1 / T" << i_R + 1 << ';';
			for (unsigned int i_N = 0; i_N < 6; ++i_N)
			{
				std::cout << std::setprecision(5) << std::scientific << "\t | \t" << T[0][i_N] / T[i_R][i_N];
				csv << T[0][i_N] / T[i_R][i_N] << ';';
			}
		}
	}
	csv.close();
}