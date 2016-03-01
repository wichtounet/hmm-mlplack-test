#include <mlpack/core.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/gmm/gmm.hpp>

using GMM = mlpack::gmm::GMM;

template<typename Distribution>
using HMM = mlpack::hmm::HMM<Distribution>;

using GaussianDistribution = mlpack::distribution::GaussianDistribution;

int main(){
    srand(time(NULL));

    // We will use two GMMs; one with two components and one with three.
    std::vector<GMM> gmms(2, GMM(2, 2));
    gmms[0].Weights() = arma::vec("0.3 0.7");

    // N([2.25 3.10], [1.00 0.20; 0.20 0.89])
    gmms[0].Component(0) = GaussianDistribution("4.25 3.10",
        "1.00 0.20; 0.20 0.89");

    // N([4.10 1.01], [1.00 0.00; 0.00 1.01])
    gmms[0].Component(1) = GaussianDistribution("7.10 5.01",
        "1.00 0.00; 0.00 1.01");

    gmms[1].Weights() = arma::vec("0.20 0.80");

    gmms[1].Component(0) = GaussianDistribution("-3.00 -6.12",
        "1.00 0.00; 0.00 1.00");

    gmms[1].Component(1) = GaussianDistribution("-4.25 -2.12",
        "1.50 0.60; 0.60 1.20");

    // Transition matrix.
    arma::mat transMat("0.40 0.60;"
        "0.60 0.40");

    // Make a sequence of observations.
    std::vector<arma::mat> observations(5, arma::mat(2, 2500));
    std::vector<arma::Row<size_t> > states(5, arma::Row<size_t>(2500));
    for (size_t obs = 0; obs < 5; obs++){
        states[obs][0] = 0;
        observations[obs].col(0) = gmms[0].Random();

        for (size_t i = 1; i < 2500; i++){
            double randValue = (double) rand() / (double) RAND_MAX;

            if (randValue <= transMat(0, states[obs][i - 1])){
                states[obs][i] = 0;
            } else {
                states[obs][i] = 1;
            }

            observations[obs].col(i) = gmms[states[obs][i]].Random();
        }
    }

    // Set up the GMM for training.
    HMM<GMM> hmm(2, GMM(2, 2));

    // Train the HMM.
    hmm.Train(observations, states);

    std::cout << hmm.LogLikelihood(observations[0]) << std::endl;
    std::cout << hmm.LogLikelihood(observations[1]) << std::endl;

    // Now the emission probabilities (the GMMs).
    // We have to sort each GMM for comparison.
    arma::uvec sortedIndices = sort_index(hmm.Emission()[0].Weights());

    // Sort the GMM.
    sortedIndices = sort_index(hmm.Emission()[1].Weights());
}
