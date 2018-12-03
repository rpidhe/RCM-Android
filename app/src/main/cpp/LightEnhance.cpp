#include"LightEnhance.h"
#include "Eigen/dense"
#include "Eigen/sparse"
#include<cmath>
#include<vector>
void matrixPow(float* m1Data, float* m2Data,float* dst, int size)
{
	for (int i = 0; i < size; i++)
	{
		dst[i] = pow(m1Data[i], m2Data[i]);
	}
}
void copyRGB(float* rgbData, float* rgbaData,
			 int rows, int cols)
{
	int size = rows*cols*4;
	for(int i = 0;i<rows;i++)
	{
		int idx = i * cols;
		for(int j = 0;j<cols;j++)
		{
			int rgba_j = 4 * (idx + j);
			int rgb_j = 3 * (idx + j);
			for(int k = 0;k<3;k++)
			{
				rgbaData[rgba_j + k] = rgbData[rgb_j+k];
			}
		}
	}
}
typedef Eigen::Triplet<double> T;
void spdiags(std::vector<T>& tripletList, float* data, int len,int diag, int k,bool sym = true)
{
	if(diag <= 0)
		for (int j = 0; j < len; j++)
		{
			int i = -diag + j;
			if (i >= k)break;
			if (data[j] != 0)
			{
				tripletList.push_back(T(i, j, data[j]));
				if(sym)
					tripletList.push_back(T(j, i, data[j]));
			}

		}
	else
		for (int i = 0; i < len; i++)
		{
			int j = diag + i;
			if (j >= k)break;
			if (data[i] != 0)
			{
				tripletList.push_back(T(i, j, data[i]));
				if (sym)
					tripletList.push_back(T(j, i, data[j]));
			}

		}

}
Eigen::VectorXf solveLinearEquation(Eigen::MatrixXf& t0, Eigen::MatrixXf& wx, Eigen::MatrixXf& wy,float lambda,bool need_pcg = false)
{
	int c = t0.cols(), r = t0.rows();
	int k = r*c;
	Eigen::MatrixXf dxa(r, c), dya(r, c);
	for (int i = 1; i < r; i++)
	{
		dya.row(i) = wy.row(i - 1) + wy.row(i);
	}
	dya.row(0) = wy.row(r - 1) + wy.row(0);
	for (int i = 1; i < c; i++)
	{
		dxa.col(i) = wx.col(i - 1) + wx.col(i);
	}
	dxa.col(0) = wx.col(c - 1) + wx.col(0);
	Eigen::MatrixXf D = ((dxa + dya).array()*lambda + 1);

	std::vector<T> tripletList;
	tripletList.reserve(5 * k);

	wx *= -lambda;
	spdiags(tripletList, wx.col(c - 1).data(), r, -k + r, k);
	wx.col(c - 1).setZero();
	spdiags(tripletList, wx.data(), k - r, -r, k);

	wy *= -lambda;
	Eigen::MatrixXf dyd1(r, c);
	dyd1.setZero();
	dyd1.row(0) = wy.row(r - 1);
	spdiags(tripletList, dyd1.data(), k, -r + 1, k);
	wy.row(r - 1).setZero();
	spdiags(tripletList, wy.data(), k, -1, k);

	spdiags(tripletList, D.data(), k, 0, k,false);

	Eigen::SparseMatrix<float> A(k, k);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();

	Eigen::Map<Eigen::VectorXf> b(t0.data(), k, 1);
	//Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Upper> solver;
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Upper, Eigen::IncompleteCholesky<float>> solver;
	solver.setMaxIterations(50);
	solver.setTolerance(0.1f);

	Eigen::VectorXf x = solver.compute(A).solve(b);
	return x;
}

void getK(float* tData, float* outData, int h, int w)
{
	int win = 5;
	float alpha = 1;
	bool need_pcg = true;
	float eps = 0.001;
	Eigen::MatrixXf t0 = Eigen::Map<Eigen::MatrixXf>(tData,h,w);
	Eigen::MatrixXf dt0_v(h, w), dt0_h(h,w);
	for (int i = 0; i < h; i++)
	{
		dt0_v.row(i) = t0.row((i+1)%h) - t0.row(i);
	}
	for (int i = 0; i < w; i++)
	{
		dt0_h.col(i) = t0.col((i + 1) % w) - t0.col(i);
	}
	Eigen::MatrixXf gauker_v(h, w), gauker_h(h, w);
	int halfwin = win / 2;
	for (int i = 0; i < h; i++)
	{
		gauker_v.row(i).setZero();
		for (int k = -halfwin; k <= halfwin; k++)
		{
			int r = i + k;
			if (r >= 0 && r < h)
				gauker_v.row(i) += dt0_v.row(r);
		}
	}
	for (int j = 0; j < w; j++)
	{
		gauker_h.col(j).setZero();
		for (int k = -halfwin; k <= halfwin; k++)
		{
			int r = j + k;
			if (r >= 0 && r<w)
				gauker_h.col(j) += dt0_h.col(r);
		}
	}
	Eigen::MatrixXf W_h = (gauker_h.array().abs().cwiseProduct(dt0_h.array().abs()) + eps).cwiseInverse();
	Eigen::MatrixXf W_v = (gauker_v.array().abs().cwiseProduct(dt0_v.array().abs()) + eps).cwiseInverse();
	Eigen::VectorXf out  = solveLinearEquation(t0, W_h, W_v, alpha / 2, need_pcg);
	memcpy(outData, out.data(), w*h*sizeof(float));
}
