#include <mlpack/core.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/gmm/gmm.hpp>

using GMM = mlpack::gmm::GMM;

template<typename Distribution>
using HMM = mlpack::hmm::HMM<Distribution>;

using GaussianDistribution = mlpack::distribution::GaussianDistribution;

int main(){
    //Number of gaussians
    static constexpr const std::size_t n_gaussians = 2;

    //Number of states
    static constexpr const std::size_t n_states = 2;

    //Number of features
    static constexpr const std::size_t n_features = 2;

    std::vector<arma::mat> images;
    std::vector<arma::Row<size_t>> labels;

    {
        images.emplace_back(n_features, 6);
        labels.emplace_back(6);
        auto& image = images.back();
        auto& label = labels.back();

        image.col(0)[0] = 44;
        image.col(0)[1] = 11;
        image.col(1)[0] = 55;
        image.col(1)[1] = 10;
        image.col(2)[0] = 23;
        image.col(2)[1] = 12;
        image.col(3)[0] = 11; image.col(3)[1] = 9;
        image.col(4)[0] = 20;
        image.col(4)[1] = 8;
        image.col(5)[0] = 40;
        image.col(5)[1] = 5;

        label = {0, 1, 1, 0, 0, 1};
    }

    {
        images.emplace_back(n_features, 6);
        labels.emplace_back(6);
        auto& image = images.back();
        auto& label = labels.back();

        image.col(0)[0] = 43;
        image.col(0)[1] = 10;
        image.col(1)[0] = 50;
        image.col(1)[1] = 11;
        image.col(2)[0] = 22;
        image.col(2)[1] = 11;
        image.col(3)[0] = 10;
        image.col(3)[1] = 8;
        image.col(4)[0] = 22;
        image.col(4)[1] = 9;
        image.col(5)[0] = 43;
        image.col(5)[1] = 4;

        label = {0, 1, 1, 0, 0, 1};
    }

    {
        images.emplace_back(n_features, 7);
        labels.emplace_back(7);
        auto& image = images.back();
        auto& label = labels.back();

        image.col(0)[0] = 43;
        image.col(0)[1] = 10;
        image.col(1)[0] = 50;
        image.col(1)[1] = 11;
        image.col(2)[0] = 22;
        image.col(2)[1] = 11;
        image.col(3)[0] = 10;
        image.col(3)[1] = 8;
        image.col(4)[0] = 22;
        image.col(4)[1] = 9;
        image.col(5)[0] = 43;
        image.col(5)[1] = 4;
        image.col(6)[0] = 22;
        image.col(6)[1] = 3;

        label = {0, 1, 1, 0, 0, 1, 1};
    }

    {
        images.emplace_back(n_features, 6);
        labels.emplace_back(6);

        labels.back() = labels[0];
        images.back() = images[0] + 0.1;
    }

    {
        images.emplace_back(n_features, 6);
        labels.emplace_back(6);

        labels.back() = labels[1];
        images.back() = images[1] - 0.1;
    }

    // Train a HMM.

    HMM<GMM> hmm(n_states, GMM(n_gaussians, n_features));

    hmm.Train(images, labels);

    //Train a GMM directly

    GMM gmm(n_gaussians, n_features);

    gmm.Train(images[0]);

    for(std::size_t i = 1; i < images.size(); ++i){
        gmm.Train(images[i], 1, true);
    }

    // Some tests

    auto likelihood = [&gmm](auto& image){
        double likelihood = std::log(gmm.Probability(image.col(0)));

        for(std::size_t f = 1; f < image.n_cols; ++f){
            likelihood += std::log(gmm.Probability(image.col(f)));
        }

        return likelihood;
    };

    for(auto& image : images){
        std::cout << "From training(hmm): " << hmm.LogLikelihood(image) << std::endl;
        std::cout << "From training(gmm): : " << likelihood(image) << std::endl;
    }

    {
        arma::mat likely(n_features, 7);
        likely = images[2] + 1.0;
        std::cout << "Likely(hmm):" << hmm.LogLikelihood(likely) << std::endl;
        std::cout << "Likely(gmm):" << likelihood(likely) << std::endl;
    }

    {
        arma::mat likely(n_features, 7);
        likely = images[2] - 1.0;
        std::cout << "Likely(hmm):" << hmm.LogLikelihood(likely) << std::endl;
        std::cout << "Likely(gmm):" << likelihood(likely) << std::endl;
    }

    {
        arma::mat unlikely(n_features, 6);
        unlikely = images[0] * 2.0;
        std::cout << "Unlikely(hmm):" << hmm.LogLikelihood(unlikely) << std::endl;
        std::cout << "Unlikely(gmm): : " << likelihood(unlikely) << std::endl;
    }
}
