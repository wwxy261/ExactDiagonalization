#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <chrono>

#ifdef Intel_MKL
#define EIGEN_USE_MKL_ALL
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "Eigen/Dense"
typedef Eigen::SparseMatrix<double> SparseMat; 
typedef Eigen::SparseVector<double> SparseVec;
typedef Eigen::Triplet<double> T;

#ifdef _ARPACK
#include "unsupported/Eigen/ArpackSupport"
typedef Eigen::SimplicialLDLT<SparseMat> SparseChol;
typedef Eigen::ArpackGeneralizedSelfAdjointEigenSolver <SparseMat, SparseChol> Arpack;

#elif defined(SPECTRA)
#include "Spectra/SymEigsSolver.h"
#include "Spectra/MatOp/SparseSymMatProd.h"
using namespace Spectra;
#endif

// System
typedef std::vector<u_int32_t> Uint32Array;
typedef std::chrono::system_clock::time_point time_point;

// Hopping 
// (idx of siteA, idx of siteB, value of hopping)
typedef std::tuple<u_int32_t, u_int32_t, double> Hopping;
typedef std::vector<Hopping> Hoppings;

// index of state Map
typedef std::unordered_map<u_int32_t, u_int32_t> StateIdxMap;

// EigenValue and its corresponding EigenVector
struct EigVal {
    double EigenVal;
    Eigen::VectorXcd EigenVec;
};


const int L = 12;
const int Nup = 6;
const int Ndown = 6;
const double U = 10.0;
const double t = -2;

int numberOf1(u_int32_t n) {
    int cnt = 0;
    while (n != 0) {
        cnt++;
        n &= (n - 1);
    }
    return cnt;
}

uint32_t factorial(uint32_t n) {
    uint32_t res = 1;
    while (n != 0) {
        res *= (n--);
    }
    return res;
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

Uint32Array getStateWithFixSpinNumber(u_int32_t n) {
    Uint32Array states;
    states.reserve(1 << L);
    for (unsigned int i = 0; i < (1 << L); i++) {
        if (numberOf1(i) == n) {
            states.emplace_back(i);
        }
    }
    return states;
}

Uint32Array getStateWithFixNumber(Uint32Array& spinUpStates, Uint32Array& spinDownStates) {
    Uint32Array states;
    states.reserve(1 << L);
    for (auto spinUpState : spinUpStates) {
        for (auto spinDownState : spinDownStates) {
            states.emplace_back((spinUpState << L) + spinDownState);
        }
    }
    return states;
}

StateIdxMap getStateIdxMap(Uint32Array states) {
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

SparseMat Hamiltonian(Uint32Array& states, StateIdxMap& stateIdxMap, std::vector<Hopping>& hoppings) {
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

EigVal GSLanczos(SparseMat &H, const u_int32_t size, const double eps = 1e-4, const int maxStep = 1000, const int K = 30) {

    std::vector<double> Kdiag;
    Kdiag.reserve(maxStep);
    std::vector<double> Ksub;
    Ksub.reserve(maxStep);

    double minValue = 0;
    double lastValue = 0;
    Eigen::VectorXcd KEigVec;   

    Eigen::VectorXcd f0 = Eigen::VectorXcd::Random(size);
    f0 /= f0.norm();
    Eigen::VectorXcd psi0 = f0;
    Eigen::VectorXcd f1 = H * f0;
    int step = 0;
    for (step = 0; step < maxStep; step++ ) {
        double a = f0.dot(f1).real();
        f1 = f1 - a * f0;
        double b = f1.norm();
        f1 = f1 / b;

        Eigen::VectorXcd tmp = f0;
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
        double a = f0.dot(f1).real();
        f1 = f1 - a * f0;
        double b = f1.norm();
        f1 = f1 / b;
        groundState += KEigVec[i] * f1;

        Eigen::VectorXcd tmp = f0;
        f0 = f1;
        f1 = H * f1 - b * tmp;
    }
    return {minValue, f0};
}

int main()
{
    // Constructing State of Hilbert Space with Fix N and Sz
    std::cout <<"Start Constructing State" << std::endl;
    time_point t0 = std::chrono::system_clock::now();

    Uint32Array spinUPStates = getStateWithFixSpinNumber(Nup);
    Uint32Array spinDownStates = getStateWithFixSpinNumber(Ndown);
    Uint32Array states = getStateWithFixNumber(spinUPStates, spinDownStates);
    StateIdxMap statesIdxMap = getStateIdxMap(states);

    time_point t1 = std::chrono::system_clock::now();
    printTimeCost("Constructing State", t0, t1);
    std::cout << "All  States Num: " << (1 << (2 * L)) << std :: endl;
    std::cout << "FixN States Num: " << states.size() << std::endl; 
    std::cout << "-----------------------------" << std::endl;
    
    // Add hopping in real space
    Hoppings hoppings = getNNHoppings1D(L);

    // Constructing Hamiltonian
    std::cout << "Start Constructing Hamiltonian" << std::endl;
    t0 = std::chrono::system_clock::now();
    SparseMat H = Hamiltonian(states, statesIdxMap, hoppings);
    t1 = std::chrono::system_clock::now();
    printTimeCost("Constructing Hamiltonian", t0, t1);
    std::cout << "-----------------------------" << std::endl;

    // Solve Ground State of Hamiltonian
    std::cout << "Start Solve GS of Hamiltonian" << std::endl;
    t0 = std::chrono::system_clock::now();
    double GEnergy = 0;
    Eigen::VectorXcd GState;

    #ifdef _ARPACK
    Arpack arpack;
    arpack.compute(H, 1, "SM");
    GEnergy = arpack.eigenvalues()[0];

    #elif defined(SPECTRA)
    SparseSymMatProd<double> op(H);
    SymEigsSolver<SparseSymMatProd<double>> eigs(op, 1, 4);
    eigs.init();
    int nconv = eigs.compute(SortRule::SmallestAlge);
    if (eigs.info() == CompInfo::Successful) {
        GEnergy = eigs.eigenvalues()[0];
    }
    #else
    EigVal eigVal = GSLanczos(H, states.size());
    GEnergy = eigVal.EigenVal;
    GState = eigVal.EigenVec;
    #endif
    t1 = std::chrono::system_clock::now();
    std::cout << "Ground Energy: " << GEnergy << std::endl;
    printTimeCost("GS Solve", t0, t1);
}


