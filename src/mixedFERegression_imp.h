#ifndef __MIXEDFEREGRESSION_IMP_HPP__
#define __MIXEDFEREGRESSION_IMP_HPP__

#include <iostream>
#include <random>
#include "timing.h"
#include <fstream>
#include <chrono>

#include "R_ext/Print.h"

//#include <libseq/mpi.h>
#include "../inst/include/dmumps_c.h"
#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

template<typename InputHandler, typename Integrator, UInt ORDER>
MixedFERegression<InputHandler,Integrator,ORDER>::MixedFERegression(const MeshHandler<ORDER>& mesh, const InputHandler& regressionData):
	mesh_(mesh),
	regressionData_(regressionData),
	isWTWfactorized_(false),
	isRcomputed_(false) {};

template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::addDirichletBC()
{
	UInt id1,id3;

	UInt nnodes = mesh_.num_nodes();

	const std::vector<UInt>& bc_indices = regressionData_.getDirichletIndices();
	const std::vector<Real>& bc_values = regressionData_.getDirichletValues();
	UInt nbc_indices = bc_indices.size();

	Real pen=10e20;

	for( auto i=0; i<nbc_indices; i++)
	 {
			id1=bc_indices[i];
			id3=id1+nnodes;

			A_.coeffRef(id1,id1)=pen;
			A_.coeffRef(id3,id3)=pen;


			_b(id1)+=bc_values[i]*pen;
			_b(id3)=0;
	 }

	A_.makeCompressed();
}



template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::getRightHandData(VectorXr& rightHandData)
{
	UInt nnodes = mesh_.num_nodes();
	UInt nlocations = regressionData_.getNumberofObservations();
	rightHandData = VectorXr::Zero(nnodes);

	if(regressionData_.getCovariates().rows() == 0)
	{
		if(regressionData_.isLocationsByNodes())
		{

			for (auto i=0; i<nlocations;++i)
			{
				auto index_i = regressionData_.getObservationsIndices()[i];
				rightHandData(index_i) = regressionData_.getObservations()[i];
			}
		}
		else
		{
			rightHandData=psi_.transpose()*regressionData_.getObservations();
		}
	}
	else
	{
		rightHandData=psi_.transpose()*LeftMultiplybyQ(regressionData_.getObservations());
	}
}

template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::computeDegreesOfFreedom(UInt output_index, Real lambda)
{
	int GCVmethod = regressionData_.getGCVmethod();
	switch (GCVmethod) {
		case 1:
			computeDegreesOfFreedomExact(output_index, lambda);
			break;
		case 2:
			computeDegreesOfFreedomStochastic(output_index, lambda);
			break;
	}
}

template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::computeDegreesOfFreedomExact(UInt output_index, Real lambda)
{
	timer clock;
	clock.start();

	UInt nnodes = mesh_.num_nodes();
	UInt nlocations = regressionData_.getNumberofObservations();
	Real degrees=0;

	// Case 1: MUMPS
	if (regressionData_.isLocationsByNodes() && regressionData_.getCovariates().rows() == 0 )
	{
		auto k = regressionData_.getObservationsIndices();
		DMUMPS_STRUC_C id;
		int myid, ierr;
        int argc=0;
        char ** argv= NULL;
        //MPI_Init(&argc,&argv);
		//ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

		id.sym=0;
		id.par=1;
		id.job=JOB_INIT;
		id.comm_fortran=USE_COMM_WORLD;
		dmumps_c(&id);

		std::vector<int> irn;
		std::vector<int> jcn;
		std::vector<double> a;
		std::vector<int> irhs_ptr;
		std::vector<int> irhs_sparse;
		double* rhs_sparse= (double*)malloc(nlocations*sizeof(double));
		
		//if( myid==0){
			id.n=2*nnodes;
			for (int j=0; j<A_.outerSize(); ++j){
				for (SpMat::InnerIterator it(A_,j); it; ++it){
					irn.push_back(it.row()+1);
					jcn.push_back(it.col()+1);
					a.push_back(it.value());
				}
			}
		//}
		id.nz=irn.size();
		id.irn=irn.data();
		id.jcn=jcn.data();
		id.a=a.data();
		id.nz_rhs=nlocations;
		id.nrhs=2*nnodes;
		int j = 1;
		irhs_ptr.push_back(j);
		for (int l=0; l<k[0]-1; ++l) {
			irhs_ptr.push_back(j);
		}
		for (int i=0; i<k.size()-1; ++i) {
			++j;
			for (int l=0; l<k[i+1]-k[i]; ++l) {
				irhs_ptr.push_back(j);
			}
			
		}
		++j;
		for (int i=k[k.size()-1]; i < id.nrhs; ++i) {
			irhs_ptr.push_back(j);
		}
		for (int i=0; i<nlocations; ++i){
			irhs_sparse.push_back(k[i]+1);
		}
		id.irhs_sparse=irhs_sparse.data();
		id.irhs_ptr=irhs_ptr.data();
		id.rhs_sparse=rhs_sparse;

		#define ICNTL(I) icntl[(I)-1]
		//Output messages suppressed
		id.ICNTL(1)=-1;
		id.ICNTL(2)=-1;
		id.ICNTL(3)=-1;
		id.ICNTL(4)=0;
		id.ICNTL(20)=1;
		id.ICNTL(30)=1;
		id.ICNTL(14)=200;

		id.job=6;
		dmumps_c(&id);
		id.job=JOB_END;
		dmumps_c(&id);

		//if (myid==0){
			for (int i=0; i< nlocations; ++i){
				//std::cout << "rhs_sparse" << rhs_sparse[i] << std::endl;
				degrees+=rhs_sparse[i];
			}
		//}
		free(rhs_sparse);

		//MPI_Finalize();
	}
	// Case 2: Eigen
	else{
		MatrixXr X1 = psi_.transpose() * LeftMultiplybyQ(psi_);

		if (isRcomputed_ == false ){
			isRcomputed_ = true;
			Eigen::SparseLU<SpMat> solver;
			solver.compute(R0_);
			auto X2 = solver.solve(R1_);
			R_ = R1_.transpose() * X2;
		}

		MatrixXr X3 = X1 + lambda * R_;
		Eigen::LLT<MatrixXr> Dsolver(X3);

		auto k = regressionData_.getObservationsIndices();

		if(regressionData_.isLocationsByNodes() && regressionData_.getCovariates().rows() != 0) {
			degrees += regressionData_.getCovariates().cols();

			// Setup rhs B
			MatrixXr B;
			B = MatrixXr::Zero(nnodes,nlocations);
			// B = I(:,k) * Q
			for (auto i=0; i<nlocations;++i) {
				VectorXr ei = VectorXr::Zero(nlocations);
				ei(i) = 1;
				VectorXr Qi = LeftMultiplybyQ(ei);
				for (int j=0; j<nlocations; ++j) {
					B(k[i], j) = Qi(j);
				}
			}
			// Solve the system TX = B
			MatrixXr X;
			X = Dsolver.solve(B);
			// Compute trace(X(k,:))
			for (int i = 0; i < k.size(); ++i) {
				degrees += X(k[i], i);
			}
		}

		if (!regressionData_.isLocationsByNodes()){
			MatrixXr X;
			X = Dsolver.solve(MatrixXr(X1));

			if (regressionData_.getCovariates().rows() != 0) {
				degrees += regressionData_.getCovariates().cols();
			}
			for (int i = 0; i<nnodes; ++i) {
				degrees += X(i,i);
			}
		}
	}
	_dof[output_index] = degrees;
	_var[output_index] = 0;
	
	std::cout << "Time required for GCV computation" << std::endl;
	clock.stop();
}

template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::computeDegreesOfFreedomStochastic(UInt output_index, Real lambda)
{
	timer clock1;
	clock1.start();
	UInt nnodes = mesh_.num_nodes();
	UInt nlocations = regressionData_.getNumberofObservations();

	std::default_random_engine generator;
	// Set the initial state of the random number generator
	if (regressionData_.getRNGstate() != "") {
	  //std::cout << "SETTING RNG STATE: " << std::endl;
		std::stringstream initialRNGstate;
		initialRNGstate << regressionData_.getRNGstate();
		initialRNGstate >> generator;
	}
	// Creation of the random matrix
	std::bernoulli_distribution distribution(0.5);
	UInt nrealizations = regressionData_.getNrealizations();
	MatrixXr u(nlocations, nrealizations);
	for (int j=0; j<nrealizations; ++j) {
		for (int i=0; i<nlocations; ++i) {
			if (distribution(generator)) {
				u(i,j) = 1.0;
			}
			else {
				u(i,j) = -1.0;
			}
		}
	}
	// Save state of random number generator
	std::stringstream finalRNGstate;
	finalRNGstate << generator;
	finalRNGstate >> _finalRNGstate;

	// Define the first right hand side : | I  0 |^T * psi^T * Q * u
	MatrixXr b = MatrixXr::Zero(2*nnodes,u.cols());
	b.topRows(nnodes) = psi_.transpose()* LeftMultiplybyQ(u);

	// Resolution of the system
	Eigen::SparseLU<SpMat> solver;
	solver.compute(_coeffmatrix);
	auto x = solver.solve(b);

	MatrixXr uTpsi = u.transpose()*psi_;
	VectorXr edf_vect(nrealizations);
	Real q = 0;
	Real var = 0;

	// Degrees of freedom = q + E[ u^T * psi * | I  0 |* x ]
	if (regressionData_.getCovariates().rows() != 0) {
		q = regressionData_.getCovariates().cols();
	}
	// For any realization we calculate the degrees of freedom
	for (int i=0; i<nrealizations; ++i) {
		edf_vect(i) = uTpsi.row(i).dot(x.col(i).head(nnodes)) + q;
		var += edf_vect(i)*edf_vect(i);
	}

	// Estimates: sample mean, sample variance
	Real mean = edf_vect.sum()/nrealizations;
	_dof[output_index] = mean;
	var /= nrealizations;
	var -= mean*mean;
	_var[output_index]=var;


	std::cout << "Time required to calculate the GCV" << std::endl;
	//clock1.stop();
	timespec time;
	time = clock1.stop();
	_time[output_index]= (long long)time.tv_sec + (double)time.tv_nsec/1000000000;
}



//Implementation kept from Sangalli et al
template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::smoothLaplace()
{
	//std::cout<<"Laplace Penalization - Order: "<<ORDER<<std::endl;

	//UInt ndata=regressionData_.getObservations().size();	
	
	UInt nnodes=mesh_.num_nodes();

	FiniteElement<Integrator, ORDER> fe;

	typedef EOExpr<Mass> ETMass;
	typedef EOExpr<Stiff> ETStiff;

	Mass EMass;
	Stiff EStiff;

	ETMass mass(EMass);
	ETStiff stiff(EStiff);


	setPsi();
 	
 
 	if(!regressionData_.getCovariates().rows() == 0)
 	{
 		setH();
 		setQ();
 	}
 
     if(!regressionData_.isLocationsByNodes())
     {
     	getDataMatrix(DMat_);
     }
     else
     {
     	getDataMatrixByIndices(DMat_);
     }






	Assembler::operKernel(stiff, mesh_, fe, R1_);
	Assembler::operKernel(mass, mesh_, fe, R0_);

	VectorXr rightHandData;
	getRightHandData(rightHandData);
	_b = VectorXr::Zero(2*nnodes);
	_b.topRows(nnodes)=rightHandData;

	// timer clock;
	// clock.start();
	// Eigen::SparseLU<SpMat> solver;
 //  	solver.compute(R0_);
 //  	std::cout << "Factorized R0" << std:: endl;
 // 	MatrixXr X2 = solver.solve(R1_);
 //  	std::cout << "Inverted" << std:: endl;
 //  	MatrixXr X3 = R1_.transpose() * X2;
 //  	std::cout << "Creating R: time" << std::endl;
 //  	clock.stop();

	_solution.resize(regressionData_.getLambda().size());
	_dof.resize(regressionData_.getLambda().size());
	_var.resize(regressionData_.getLambda().size());
	_time.resize(regressionData_.getLambda().size());
	
	for(UInt i = 0; i<regressionData_.getLambda().size(); ++i)
	{

		Real lambda = regressionData_.getLambda()[i];
		SpMat R1_lambda = (-lambda) * R1_;
		SpMat R0_lambda = (-lambda)*R0_;
		this->buildA(psi_, R1_lambda, R0_lambda);


		this->buildCoeffMatrix(DMat_, R1_lambda, R0_lambda);



		//Appling border conditions if necessary
		if(regressionData_.getDirichletIndices().size() != 0)
			addDirichletBC();

		std::cout << "System resolution: time" << std::endl;
		timer clock;
		timespec time;
		clock.start();
		system_factorize();
		_solution[i] = this->template system_solve(this->_b);
		time = clock.stop();
		//_time[i]= (long long)time.tv_sec + (double)time.tv_nsec/1000000000;

		if(regressionData_.computeDOF())
			computeDegreesOfFreedom(i, lambda);
		else
			_dof[i] = -1;

		//double ISE;
		//auto f = _solution[i].segment(0,X3.rows());
		//ISE = f.transpose()*X3*f;		
		//std::cout << "asd The ISE for lambda = " << lambda << " is " << ISE << std::endl;
	}

}

//Implementation kept from Sangalli et al
template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::smoothEllipticPDE()
{
	//std::cout<<"Elliptic PDE Penalization - Order: "<<ORDER<<std::endl;
	UInt nnodes=mesh_.num_nodes();

	FiniteElement<Integrator, ORDER> fe;

	typedef EOExpr<Mass> ETMass;
	typedef EOExpr<Stiff> ETStiff;
	typedef EOExpr<Grad> ETGrad;

	Mass EMass;
	Stiff EStiff;
	Grad EGrad;

	ETMass mass(EMass);
	ETStiff stiff(EStiff);
	ETGrad grad(EGrad);

	setPsi();

	const Real& c = regressionData_.getC();
	const Eigen::Matrix<Real,2,2>& K = regressionData_.getK();
	const Eigen::Matrix<Real,2,1>& beta = regressionData_.getBeta();
	Assembler::operKernel(c*mass+stiff[K]+dot(beta,grad), mesh_, fe, R1_);
	Assembler::operKernel(mass, mesh_, fe, R0_);

	VectorXr rightHandData;
	getRightHandData(rightHandData);
	_b = VectorXr::Zero(2*nnodes);
	_b.topRows(nnodes)=rightHandData;

	_solution.resize(regressionData_.getLambda().size());
	_dof.resize(regressionData_.getLambda().size());
	_var.resize(regressionData_.getLambda().size());
	_time.resize(regressionData_.getLambda().size());

	/*timer clock;
	clock.start();
	Eigen::SparseLU<SpMat> solver;
  	solver.compute(R0_);
  	std::cout << "Factorized R0" << std:: endl;
 	MatrixXr X2 = solver.solve(R1_);
  	std::cout << "Inverted" << std:: endl;
  	MatrixXr X3 = R1_.transpose() * X2;
  	std::cout << "Creating R: time" << std::endl;
  	clock.stop();*/

	for(UInt i = 0; i<regressionData_.getLambda().size(); ++i)
	{

		Real lambda = regressionData_.getLambda()[i];
		SpMat R1_lambda = (-lambda)*R1_;
		SpMat R0_lambda = (-lambda)*R0_;
		this->buildA(psi_, R1_lambda, R0_lambda);

		//Appling border conditions if necessary
		if(regressionData_.getDirichletIndices().size() != 0)
			addDirichletBC();

		std::cout << "System resolution: time" << std::endl;
		timer clock;
		timespec time;
		clock.start();
		system_factorize();
		_solution[i] = this->template system_solve(this->_b);
		time = clock.stop();
		_time[i]= (long long)time.tv_sec + (double)time.tv_nsec/1000000000;

		if(regressionData_.computeDOF())
			computeDegreesOfFreedom(i, lambda);
		else
			_dof[i] = -1;

		// double ISE;
		//auto f = _solution[i].segment(0,X3.rows());
		//ISE = f.transpose()*X3*f;		
		//std::cout << "asd The ISE for lambda = " << lambda << " is " << ISE << std::endl;
	}

}


//Implementation kept from Sangalli et al
template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::smoothEllipticPDESpaceVarying()
{
	//std::cout<<"Space-varying Coefficient - Elliptic PDE Penalization - Order: "<<ORDER<<std::endl;

	UInt nnodes=mesh_.num_nodes();

	FiniteElement<Integrator, ORDER> fe;

	typedef EOExpr<Mass> ETMass;
	typedef EOExpr<Stiff> ETStiff;
	typedef EOExpr<Grad> ETGrad;

	Mass EMass;
	Stiff EStiff;
	Grad EGrad;

	ETMass mass(EMass);
	ETStiff stiff(EStiff);
	ETGrad grad(EGrad);

	setPsi();

	const Reaction& c = regressionData_.getC();
	const Diffusivity& K = regressionData_.getK();
	const Advection& beta = regressionData_.getBeta();
	Assembler::operKernel(c*mass+stiff[K]+dot(beta,grad), mesh_, fe, R1_);
	Assembler::operKernel(mass, mesh_, fe, R0_);

	const ForcingTerm& u = regressionData_.getU();
	VectorXr forcingTerm;
	Assembler::forcingTerm(mesh_,fe, u, forcingTerm);

	VectorXr rightHandData;
	getRightHandData(rightHandData);
	//_b.resize(2*nnodes);
	_b = VectorXr::Zero(2*nnodes);
	_b.topRows(nnodes)=rightHandData;
	//std::cout<<"Forcing Term "<<std::cout<<forcingTerm<<"END";
	_b.bottomRows(nnodes)=forcingTerm;

	_solution.resize(regressionData_.getLambda().size());
	_dof.resize(regressionData_.getLambda().size());
	_var.resize(regressionData_.getLambda().size());
	_time.resize(regressionData_.getLambda().size());

	// timer clock;
	// clock.start();
	// Eigen::SparseLU<SpMat> solver;
 //   	solver.compute(R0_);
 //   	std::cout << "Factorized R0" << std:: endl;
 //  	MatrixXr X2 = solver.solve(R1_);
 // 	MatrixXr X3 = R1_.transpose() * X2;
 //   	std::cout << "Creating R: time" << std::endl;
 //  	clock.stop();

	for(UInt i = 0; i<regressionData_.getLambda().size(); ++i)
	{

		Real lambda = regressionData_.getLambda()[i];
		SpMat R1_lambda = (-lambda)*R1_;
		SpMat R0_lambda = (-lambda)*R0_;
		this->buildA(psi_, R1_lambda, R0_lambda);

		//Appling border conditions if necessary
		if(regressionData_.getDirichletIndices().size() != 0)
			addDirichletBC();

		std::cout << "System resolution: time" << std::endl;
		timer clock;
		timespec time;
		clock.start();
		system_factorize();
		_solution[i] = this->template system_solve(this->_b);
		time = clock.stop();
		_time[i]= (long long)time.tv_sec + (double)time.tv_nsec/1000000000;

		if(regressionData_.computeDOF())
			computeDegreesOfFreedom(i, lambda);
		else
			_dof[i] = -1;

		// double ISE;
		// auto f = _solution[i].segment(0,X3.rows());
		// ISE = f.transpose()*X3*f;		
		// std::cout << "The ISE for lambda = " << lambda << " is " << ISE << std::endl;


	}

}


template<typename InputHandler, typename Integrator, UInt ORDER>
MatrixXr MixedFERegression<InputHandler,Integrator,ORDER>::LeftMultiplybyQ(const MatrixXr& u)
{	
	if (regressionData_.getCovariates().rows() == 0){
		return u;
	}
	else{
		MatrixXr W(this->regressionData_.getCovariates());
		if (isWTWfactorized_ == false ){
			WTWinv_.compute(W.transpose()*W);
			isWTWfactorized_=true;
		}
		MatrixXr Pu= W*WTWinv_.solve(W.transpose()*u);
		return u-Pu;
	}

}

template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::buildA(const SpMat& Psi,  const SpMat& R1,  const SpMat& R0) {

	UInt nnodes = mesh_.num_nodes();
	
	SpMat DMat = Psi.transpose()*Psi;

	std::vector<coeff> tripletAll;
	tripletAll.reserve(DMat.nonZeros() + 2*R1.nonZeros() + R0.nonZeros());

	for (int k=0; k<DMat.outerSize(); ++k)
		for (SpMat::InnerIterator it(DMat,k); it; ++it)
		{
			tripletAll.push_back(coeff(it.row(), it.col(),it.value()));
		}
	for (int k=0; k<R0.outerSize(); ++k)
		for (SpMat::InnerIterator it(R0,k); it; ++it)
		{
			tripletAll.push_back(coeff(it.row()+nnodes, it.col()+nnodes,it.value()));
		}
	for (int k=0; k<R1.outerSize(); ++k)
	  for (SpMat::InnerIterator it(R1,k); it; ++it)
	  {
		  tripletAll.push_back(coeff(it.col(), it.row()+nnodes,it.value()));
	  }
	for (int k=0; k<R1.outerSize(); ++k)
	  for (SpMat::InnerIterator it(R1,k); it; ++it)
	  {
		  tripletAll.push_back(coeff(it.row()+nnodes, it.col(), it.value()));
	  }

	A_.setZero();
	A_.resize(2*nnodes,2*nnodes);
	A_.setFromTriplets(tripletAll.begin(),tripletAll.end());
	A_.makeCompressed();
	//std::cout<<"Coefficients' Matrix Set Correctly"<<std::endl;
}

template<typename InputHandler, typename Integrator, UInt ORDER>
void MixedFERegression<InputHandler,Integrator,ORDER>::system_factorize() {

	UInt nnodes = mesh_.num_nodes();

	// First phase: Factorization of matrix A
	Adec_.compute(A_);
	
	if (regressionData_.getCovariates().rows() != 0) {
		// Second phase: factorization of matrix  G =  C + [V * A^-1 * U]

		// Definition of matrix U = [ psi * W | 0 ]^T
		MatrixXr W(this->regressionData_.getCovariates());
		U_ = MatrixXr::Zero(2*nnodes, W.cols());
		U_.topRows(nnodes) = psi_.transpose()*W;

		// D = U^T * A^-1 * U
		MatrixXr D = U_.transpose()*Adec_.solve(U_);
		// G = C + D
		MatrixXr G = -W.transpose()*W + D;
		Gdec_.compute(G);
	}
}

template<typename InputHandler, typename Integrator, UInt ORDER>
template<typename Derived>
MatrixXr MixedFERegression<InputHandler,Integrator,ORDER>::system_solve(const Eigen::MatrixBase<Derived> &b) {

	// Resolution of the system A * x1 = b
	MatrixXr x1 = Adec_.solve(b);
	
	if (regressionData_.getCovariates().rows() != 0) {
		// Resolution of G * x2 = U^T * x1
		MatrixXr x2 = Gdec_.solve(U_.transpose()*x1);
		// Resolution of the system A * x3 = U * x2
		x1 -= Adec_.solve(U_*x2);
	}

	return x1;
}











template<typename InputHandler, typename Integrator, UInt ORDER>
 void MixedFERegression<InputHandler,Integrator,ORDER>::buildCoeffMatrix(const SpMat& DMat,  const SpMat& AMat,  const SpMat& MMat)
 {
 	//I reserve the exact memory for the nonzero entries of each row of the coeffmatrix for boosting performance
 	//_coeffmatrix.setFromTriplets(tripletA.begin(),tripletA.end());
 
 	UInt nnodes = mesh_.num_nodes();
 
 	std::vector<coeff> tripletAll;
 	tripletAll.reserve(DMat.nonZeros() + 2*AMat.nonZeros() + MMat.nonZeros());
 
 	for (int k=0; k<DMat.outerSize(); ++k)
 	  for (SpMat::InnerIterator it(DMat,k); it; ++it)
 	  {
 		  tripletAll.push_back(coeff(it.row(), it.col(),it.value()));
 	  }
 	for (int k=0; k<MMat.outerSize(); ++k)
 	  for (SpMat::InnerIterator it(MMat,k); it; ++it)
 	  {
 		  tripletAll.push_back(coeff(it.row()+nnodes, it.col()+nnodes,it.value()));
 	  }
 	for (int k=0; k<AMat.outerSize(); ++k)
 	  for (SpMat::InnerIterator it(AMat,k); it; ++it)
 	  {
 		  tripletAll.push_back(coeff(it.col(), it.row()+nnodes,it.value()));
 	  }
 	for (int k=0; k<AMat.outerSize(); ++k)
 	  for (SpMat::InnerIterator it(AMat,k); it; ++it)
 	  {
 		  tripletAll.push_back(coeff(it.row()+nnodes, it.col(), it.value()));
 	  }
 
 	_coeffmatrix.setZero();
 	_coeffmatrix.resize(2*nnodes,2*nnodes);
 	_coeffmatrix.setFromTriplets(tripletAll.begin(),tripletAll.end());
 	_coeffmatrix.makeCompressed();
 	//std::cout<<"Coefficients' Matrix Set Correctly"<<std::endl;
 }

template<typename InputHandler, typename Integrator, UInt ORDER>
 void MixedFERegression<InputHandler,Integrator,ORDER>::getDataMatrix(SpMat& DMat)
 {
 		UInt nnodes = mesh_.num_nodes();
 		//UInt nlocations = regressionData_.getNumberofObservations();
 
 		DMat.resize(nnodes,nnodes);
 
 		if (regressionData_.getCovariates().rows() == 0)
 			DMat = psi_.transpose()*psi_;
 		else
 		{
 			DMat = (SpMat(psi_.transpose())*Q_*psi_).sparseView();
 		}
 }
 
 template<typename InputHandler, typename Integrator, UInt ORDER>
 void MixedFERegression<InputHandler,Integrator,ORDER>::getDataMatrixByIndices(SpMat& DMat)
 {
 		UInt nnodes = mesh_.num_nodes();
 		UInt nlocations = regressionData_.getNumberofObservations();
 
 		DMat.resize(nnodes,nnodes);
 
 		if (regressionData_.getCovariates().rows() == 0)
 		{
 			DMat.reserve(1);
 			for (auto i = 0; i<nlocations; ++i)
 			{
 				auto index = regressionData_.getObservationsIndices()[i];
 				DMat.insert(index,index) = 1;
 			}
 		}
 		else
 		{
 			//May be inefficient
 			for (auto i = 0; i<nlocations; ++i)
 			{
 				auto index_i = regressionData_.getObservationsIndices()[i];
 				for (auto j = 0; j<nlocations; ++j)
 				{
 					auto index_j = regressionData_.getObservationsIndices()[j];
 					DMat.insert(index_i,index_j) = Q_(i,j);
 				}
 			}
 		}
 }
 
 template<typename InputHandler, typename Integrator, UInt ORDER>
 void MixedFERegression<InputHandler,Integrator,ORDER>::setPsi(){/////
 
 		//std::cout<<"Data Matrix Computation by Basis Evaluation.."<<std::endl;
 		UInt nnodes = mesh_.num_nodes();
 		UInt nlocations = regressionData_.getNumberofObservations();
 		Real eps = 2.2204e-016,
 			 tolerance = 100 * eps;
 
 		//cout<<"Nodes number "<<nnodes<<"Locations number "<<nlocations<<endl;
 
 		//std::vector<coeff> entries;
 		//entries.resize((ORDER * 3)*nlocations);
 
 
 		psi_.resize(nlocations, nnodes);
 		//psi_.reserve(Eigen::VectorXi::Constant(nlocations,ORDER*3));
 
 		Triangle<ORDER*3> tri_activated;
 		Eigen::Matrix<Real,ORDER * 3,1> coefficients;
 
 		Real evaluator;
 
 		for(UInt i=0; i<nlocations;i++)
 		{
 			tri_activated = mesh_.findLocationNaive(regressionData_.getLocations()[i]);
 			if(tri_activated.getId() == Identifier::NVAL)
 			{
 				#ifdef R_VERSION_
 				Rprintf("ERROR: Point %d is not in the domain, remove point and re-perform smoothing\n", i+1);
 				#else
 				std::cout << "ERROR: Point " << i+1 <<" is not in the domain\n";
 				#endif
 			}else
 			{
 				for(UInt node = 0; node < ORDER*3 ; ++node)
 				{
 					coefficients = Eigen::Matrix<Real,ORDER * 3,1>::Zero();
 					coefficients(node) = 1; //Activates only current base
 					evaluator = evaluate_point<ORDER>(tri_activated, regressionData_.getLocations()[i], coefficients);
 					psi_.insert(i, tri_activated[node].getId()) = evaluator;
 				}
 			}
 		}
 
 		psi_.prune(tolerance);
 		psi_.makeCompressed();
 }
 
 template<typename InputHandler, typename Integrator, UInt ORDER>
 void MixedFERegression<InputHandler,Integrator,ORDER>::setQ()
 {
 	//std::cout<<"Computing Orthogonal Space Projection Matrix"<<std::endl;
 	Q_.resize(H_.rows(),H_.cols());
 	Q_ = -H_;
 	for (int i=0; i<H_.rows();++i)
 	{
 		Q_(i,i) += 1;
 	}
 }
 
 template<typename InputHandler, typename Integrator, UInt ORDER>
 void MixedFERegression<InputHandler,Integrator,ORDER>::setH()
 {
 	//std::cout<<"Computing Projection Matrix"<<std::endl;
 	//UInt nnodes = mesh_.num_nodes();
 	UInt nlocations = regressionData_.getNumberofObservations();
 
 	//regressionData_.printCovariates(std::cout);
 	MatrixXr W(this->regressionData_.getCovariates());
 	//std::cout<<"W "<< W <<std::endl;
 	//total number of mesh nodes
 	//UInt nnodes = mesh_.num_nodes();
 	if(regressionData_.isLocationsByNodes())
 	{
 		MatrixXr W_reduced(regressionData_.getNumberofObservations(), W.cols());
 		for (auto i=0; i<nlocations;++i)
 		{
 			auto index_i = regressionData_.getObservationsIndices()[i];
 			for (auto j=0; j<W.cols();++j)
 			{
 				W_reduced(i,j) = W(index_i,j);
 			}
 		}
 		W = W_reduced;
 	}
 
 
 	MatrixXr WTW(W.transpose()*W);
 
 	H_=W*WTW.ldlt().solve(W.transpose()); // using cholesky LDLT decomposition for computing hat matrix
 }
 



#endif
