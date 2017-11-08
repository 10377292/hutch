#ifndef __MIXEDFEREGRESSION_HPP__
#define __MIXEDFEREGRESSION_HPP__

#include "fdaPDE.h"
#include "finite_element.h"
#include "matrix_assembler.h"
#include "mesh.h"
#include "param_functors.h"
#include "regressionData.h"
#include "solver.h"
#include <memory>

//! A LinearSystem class: A class for the linear system construction and resolution.

template<typename InputHandler, typename Integrator, UInt ORDER>
class MixedFERegression{
	private:

		const MeshHandler<ORDER> &mesh_;
		const InputHandler& regressionData_;
		std::vector<coeff> tripletsData_;

		SpMat A_;		// System matrix with psi^T*psi in north-west block
		SpMat R1_;	// North-east block of system matrix A_
		SpMat R0_;	// South-east block of system matrix A_
		SpMat psi_;
		MatrixXr U_;	// psi^T*W padded with zeros
		
		
		Eigen::SparseLU<SpMat> Adec_; // Stores the factorization of A_
		Eigen::PartialPivLU<MatrixXr> Gdec_;	// Stores factorization of G =  C + [V * A^-1 * U]
		Eigen::PartialPivLU<MatrixXr> WTWinv_;	// Stores the factorization of W^T * W
		bool isWTWfactorized_;
		bool isRcomputed_;
		MatrixXr R_; //R1 ^T * R0^-1 * R1

		VectorXr _b;                     //!A Eigen::VectorXr: Stores the system right hand side.
		std::vector<VectorXr> _solution; //!A Eigen::VectorXr : Stores the system solution
		std::vector<Real> _dof;
		std::vector<Real> _var;
		std::string _finalRNGstate;
		std::vector<Real> _time;

		void setPsi();
		void buildA(const SpMat& Psi,  const SpMat& R1,  const SpMat& R0);
		MatrixXr LeftMultiplybyQ(const MatrixXr& u);

		SpMat DMat_;
 		SpMat AMat_;
 		SpMat MMat_;
 
 		MatrixXr Q_;
 		MatrixXr H_;
 
 		SpMat _coeffmatrix; 




 		void setQ();
 		void setH();
 
 		void buildCoeffMatrix(const SpMat& DMat,  const SpMat& AMat,  const SpMat& MMat);
 



	public:
		//!A Constructor.
		MixedFERegression(const MeshHandler<ORDER>& mesh, const InputHandler& regressionData);

		void smoothLaplace();
		void smoothEllipticPDE();
		void smoothEllipticPDESpaceVarying();

		inline std::vector<VectorXr> const & getSolution() const{return _solution;};
		inline std::vector<Real> const & getDOF() const{return _dof;};
		inline std::vector<Real> const & getVar() const{return _var;};
		inline std::vector<Real> const & getTime() const{return _time;};
		inline std::string const & getFinalRNGstate() const{return _finalRNGstate;}

		void addDirichletBC();
		void getRightHandData(VectorXr& rightHandData);
		void computeDegreesOfFreedom(UInt output_index, Real lambda);
		void computeDegreesOfFreedomExact(UInt output_index, Real lambda);
		void computeDegreesOfFreedomStochastic(UInt output_index, Real lambda);

		void system_factorize();
		template<typename Derived>
		MatrixXr system_solve(const Eigen::MatrixBase<Derived>&);

		void getDataMatrix(SpMat& DMat);
 		void getDataMatrixByIndices(SpMat& DMat);

};	

#include "mixedFERegression_imp.h"

#endif
