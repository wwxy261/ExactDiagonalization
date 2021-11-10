#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <complex>

#ifdef Intel_MKL
#define EIGEN_USE_MKL_ALL
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "Eigen/Dense"
#include "Eigen/SparseLU"
typedef Eigen::SparseMatrix<double> SparseMat; 
typedef Eigen::SparseVector<double> SparseVec;
typedef Eigen::Triplet<double> T; 

#ifdef _ARPACK
#include "unsupported/Eigen/ArpackSupport"
typedef Eigen::SimplicialLDLT<SparseMat> SparseChol;
typedef Eigen::ArpackGeneralizedSelfAdjointEigenSolver <SparseMat, SparseChol> Arpack;
#endif

// System
typedef std::vector<u_int32_t> Uint32Array;
typedef std::vector<double> DoubleArray;
typedef std::chrono::system_clock::time_point time_point;
typedef std::complex<double> Complex;

// Hopping 
// (idx of siteA, idx of siteB, value of hopping)
typedef std::tuple<u_int32_t, u_int32_t, double> Hopping;
typedef std::vector<Hopping> Hoppings;

// index of state Map
typedef std::unordered_map<u_int32_t, u_int32_t> StateIdxMap;

const int L = 12;
const int Nup = 6;
const int Ndown = 6;
const double U = 10.0;
const double t = -2;
const double delta = 0.001;
const int Nw = 1000;
Complex GF[Nw][L][L];

Hoppings hoppings;

int numberOf1(u_int32_t n) {
    int cnt = 0;
    while (n != 0) {
        cnt++;
        n &= (n - 1);
    }
    return cnt;
}

Hoppings getNNHoppings1D(int n) {
    Hoppings hoppings;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (abs(j - i) == 1) {
                hoppings.push_back({i, j, t});
            }
        }
    }
    return hoppings;
}

 Uint32Array getStatesFixSz(int nUp, int nDown) {
    auto getStatesFixNum = [](u_int32_t n) {
        Uint32Array states;
        states.reserve(1 << L);
        for (unsigned int i = 0; i < (1 << L); i++) {
            if (numberOf1(i) == n) {
                states.emplace_back(i);
            }
        }
        return states;  
    };

    Uint32Array states;
    Uint32Array spinUpStates = getStatesFixNum(nUp);
    Uint32Array spinDownStates = getStatesFixNum(nDown);
    states.reserve(1 << L);
    for (auto spinUpState : spinUpStates) {
        for (auto spinDownState : spinDownStates) {
            states.emplace_back((spinUpState << L) + spinDownState);
        }
    }
    return states;
}

StateIdxMap getStateIdxMap(Uint32Array& states) {
    StateIdxMap stateIdxMap;
    stateIdxMap.reserve(1 << L);
    for (u_int32_t i = 0; i < states.size(); i++) {
        stateIdxMap[states[i]] = i;
    }
    return stateIdxMap;
}

u_int32_t doubleOccNum(u_int32_t state) {
    u_int32_t spinUpState = state >> L;
    u_int32_t spinDownState = state & ((1 << L) - 1);
    return numberOf1(spinUpState & spinDownState);
}

int perm(u_int32_t state, u_int32_t siteA, u_int32_t siteB) {
    if (siteA < siteB) std::swap(siteA, siteB);
    return (numberOf1(state & ((1 << siteA) - 1) & ~((1 << (siteB + 1)) - 1)) % 2 ? -1 : 1);
}

SparseMat Hamiltonian(Uint32Array& states, StateIdxMap& stateIdxMap) {
    u_int32_t n = states.size();
    std::vector<T> tripletList;
    tripletList.reserve(L * n);
    for (u_int32_t i = 0; i < n; i++) {
        tripletList.emplace_back(T(i,i,U * doubleOccNum(states[i])));
        for (auto [a, b, t] : hoppings) {
            for (u_int32_t s = 0; s <= 1; s++) {
                if ((states[i] & (1 << (s * L) << b)) && !(states[i] & (1 << (s * L) << a))) {
                    u_int32_t state = states[i] ^ (1 << (s * L) << b) ^ (1 << (s * L) << a);
                    u_int32_t j = stateIdxMap[state];
                    tripletList.emplace_back(T(j, i, t * perm(states[i], a, b)));
                }
            }
        }
    }
    SparseMat H(n, n);
    H.setFromTriplets(tripletList.begin(), tripletList.end());
    return H;
}

void printTimeCost(std::string taskName, time_point t0, time_point t1) {
    std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    double second = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    std::cout << taskName << " Time Cost: " << second << " seconds." << std::endl;
}

typedef std::tuple<double, Eigen::VectorXcd> EigValueVector;
EigValueVector GSLanczos(SparseMat &H, const u_int32_t size, const double eps = 1e-4, const int maxStep = 1000, const int K = 30) {

    std::vector<double> Kdiag;
    Kdiag.reserve(maxStep);
    std::vector<double> Ksub;
    Ksub.reserve(maxStep);

    double minValue = 0;
    double lastValue = 0;
    Eigen::VectorXcd KEigVec;   

    Eigen::VectorXd f0 = Eigen::VectorXd::Random(size);
    f0 /= f0.norm();
    Eigen::VectorXd psi0 = f0;
    Eigen::VectorXd f1 = H * f0;
    int step = 0;
    for (step = 0; step < maxStep; step++ ) {
        double a = f0.dot(f1);
        f1 = f1 - a * f0;
        double b = f1.norm();
        f1 = f1 / b;

        Eigen::VectorXd tmp = f0;
        f0 = f1;
        f1 = H * f1 - b * tmp;
        Kdiag.push_back(a);

        if (step > K) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver;
            eigensolver.computeFromTridiagonal(Eigen::Map<Eigen::VectorXd>(Kdiag.data(), step), Eigen::Map<Eigen::VectorXd>(Ksub.data(), step - 1));
            minValue = eigensolver.eigenvalues()[0];
            if (abs(minValue - lastValue) < eps) {
                KEigVec = eigensolver.eigenvectors().col(0);
                break;
            }
            lastValue = minValue;
        }
        Ksub.push_back(b);
    }
    
    f0 = psi0;
    Eigen::VectorXcd groundState = KEigVec[0] * f0;
    f1 = H * f0;
    int m = KEigVec.rows();
    for (int i = 1; i < m; i++) {
        double a = f0.dot(f1);
        f1 = f1 - a * f0;
        double b = f1.norm();
        f1 = f1 / b;
        groundState += KEigVec[i] * f1;

        Eigen::VectorXd tmp = f0;
        f0 = f1;
        f1 = H * f1 - b * tmp;
    }
    return make_pair(minValue, groundState);
}

Eigen::VectorXcd createAParticleOnGS(int a, int s, Uint32Array& states, StateIdxMap & newStateIdxMap, Eigen::VectorXcd& GS) {
    int newSize = newStateIdxMap.size();
    Eigen::VectorXcd ExcitedState = Eigen::VectorXcd::Zero(newSize);
    for (int i = 0; i < states.size(); i++) {
        if ((states[i] & (1 << s << a)) == 0) {
            int newState = states[i] ^ (1 << s << a);
            int idx = newStateIdxMap[newState];
            ExcitedState[idx] = GS[i];
        }
    }
    ExcitedState /= ExcitedState.norm();
    return ExcitedState;
}

typedef std::tuple<SparseMat, Eigen::VectorXcd> GFLanczosPreRes;
GFLanczosPreRes GFLanczosPre(SparseMat& H, Eigen::VectorXcd& stateA, Eigen::VectorXcd& stateB, const int maxStep = 400) {
    int n = maxStep;
    SparseMat HKrylov(n, n);
    HKrylov.reserve(Eigen::VectorXi::Constant(n, 3));
    Eigen::VectorXcd innerProduct = Eigen::VectorXcd::Zero(n);

    Eigen::VectorXcd f0 = stateB;
    Eigen::VectorXcd f1 = H * stateB;
    for (int i = 0; i < n; i++) {
        innerProduct[i] = stateA.dot(f0);
        double a = f0.dot(f1).real();
        f1 = f1 - a * f0;
        double b = f1.norm();
        f1 = f1 / b;

        Eigen::VectorXcd tmp = f0;
        f0 = f1;
        f1 = H * f1 - b * tmp;
        HKrylov.insert(i, i) = a;
        if(i > 0) HKrylov.insert(i - 1, i) = b;
        if(i < n - 1) HKrylov.insert(i + 1, i) = b;
    }
    return std::make_tuple(HKrylov, innerProduct);
}

void calculateGF(DoubleArray& Omega, double E0, int spin, Uint32Array& states, Eigen::VectorXcd& GS) {
    int nUp = Nup, nDown = Ndown;
    spin == 0 ? nDown++ : nUp++;

    // Constructing New State and Hamiltonian in excited Space
    std::cout <<"Start Constructing State" << std::endl;
    time_point t0 = std::chrono::system_clock::now();

    Uint32Array newStates = getStatesFixSz(nUp, nDown);
    StateIdxMap newStateIdxMap = getStateIdxMap(newStates);
    SparseMat H = Hamiltonian(newStates, newStateIdxMap);

    time_point t1 = std::chrono::system_clock::now();
    printTimeCost("Constructing Excited State And Hamiltonian", t0, t1);
    std::cout << "exctied States Num: " << newStates.size() << std::endl;

    // Calculate oneParticleExStates base on GS
    std::cout <<"Start Calculate All One Particle Excite State" << std::endl;
    t0 = std::chrono::system_clock::now();
    std::vector<Eigen::VectorXcd> oneParticleExStates;
    for (int a = 0; a < L; a++) {
        oneParticleExStates.emplace_back(createAParticleOnGS(a, spin, states, newStateIdxMap, GS));
    }
    t1 = std::chrono::system_clock::now();
    printTimeCost("Calculate All One Particle Excite State", t0, t1);

    // Calculate GF now
    int nw = Omega.size();
    for (int a = 0; a < L; a++) {
        for (int b = 0; b < L; b++) {
            auto [HKrylov, innerProduct] = GFLanczosPre(H, oneParticleExStates[a], oneParticleExStates[b]);
            std::cout << "Krylov Projection Done For Site" << a << " and " << b << std::endl;
            int m = innerProduct.size();
            for (int i = 0; i < Omega.size(); i++) {
                Eigen::VectorXd e = Eigen::VectorXd::Zero(m);
                e[0] = 1.0;
                Eigen::VectorXcd X;
                Eigen::SparseLU<SparseMat, Eigen::COLAMDOrdering<int> > solver;
                // Compute the ordering permutation vector from the structural pattern of A
                solver.analyzePattern(HKrylov); 
                // Compute the numerical factorization 
                solver.factorize(HKrylov); 
                //Use the factors to solve the linear system 
                X = solver.solve(e);
                GF[i][a][b] = innerProduct.dot(X);
                std::cout << "GF Site " << a << " " << b << " Omega " << i << " Done" <<std::endl;  
            }
        }
    }
}

int main()
{
    // Constructing State of Hilbert Space with Fix N and Sz
    std::cout <<"Start Constructing State" << std::endl;
    time_point t0 = std::chrono::system_clock::now();

    Uint32Array states = getStatesFixSz(Nup, Ndown);
    StateIdxMap statesIdxMap = getStateIdxMap(states);

    time_point t1 = std::chrono::system_clock::now();
    printTimeCost("Constructing State", t0, t1);
    std::cout << "All  States Num: " << (1 << (2 * L)) << std :: endl;
    std::cout << "FixN States Num: " << states.size() << std::endl; 
    std::cout << "-----------------------------" << std::endl;
    
    // Add hopping in real space
    hoppings = getNNHoppings1D(L);

    // Constructing Hamiltonian
    std::cout << "Start Constructing Hamiltonian" << std::endl;
    t0 = std::chrono::system_clock::now();
    SparseMat H = Hamiltonian(states, statesIdxMap);
    t1 = std::chrono::system_clock::now();
    printTimeCost("Constructing Hamiltonian", t0, t1);
    std::cout << "-----------------------------" << std::endl;

    // Solve Ground State of Hamiltonian
    std::cout << "Start Solve GS of Hamiltonian" << std::endl;
    t0 = std::chrono::system_clock::now();
    double E0 = 0.0;
    Eigen::VectorXcd GS;
    #ifdef _ARPACK
    Arpack arpack;
    arpack.compute(H, 1, "SM");
    GEnergy = arpack.eigenvalues()[0];
    #else
    std::tie(E0, GS) = GSLanczos(H, states.size());
    #endif
    t1 = std::chrono::system_clock::now();
    std::cout << "Ground Energy: " << E0 << std::endl;
    printTimeCost("GS Solve", t0, t1);
    std::cout << "-----------------------------" << std::endl;

    // Solve GreenFunction of Hamiltonian
    std::cout << "Start Solve GF of Hamiltonian" << std::endl;
    t0 = std::chrono::system_clock::now();
    DoubleArray Omega = {0.0};
    // calculateGF(Omega, E0, 0, states, GS);
    t1 = std::chrono::system_clock::now();
    std::cout << "GF: " << std::endl;
    printTimeCost("Green Function Solve", t0, t1);
}


