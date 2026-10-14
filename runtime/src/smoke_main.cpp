#include "WavWriter.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::filesystem::path output = argc > 1 ? argv[1] : "smoke-cpp.wav";
    constexpr int sampleRate = 48000;
    constexpr int channels = 2;
    std::vector<float> samples(sampleRate * channels);
    for (int frame = 0; frame < sampleRate; ++frame) {
        const float value = 0.15f * std::sin(
            2.0f * std::numbers::pi_v<float> * 440.0f * static_cast<float>(frame) / sampleRate);
        samples[frame * channels] = value;
        samples[frame * channels + 1] = value;
    }

    std::string error;
    if (!acestep::writeFloatWav(output, samples, sampleRate, channels, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "wrote " << std::filesystem::absolute(output).string() << '\n';
    return 0;
}

