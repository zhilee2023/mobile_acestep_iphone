#include "WavWriter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>

namespace acestep {
namespace {

template <typename T>
void writeValue(std::ofstream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

}  // namespace

bool writeFloatWav(
    const std::filesystem::path& path,
    std::span<const float> samples,
    int sampleRate,
    int channels,
    std::string& error) {
    if (sampleRate <= 0 || channels <= 0 || samples.size() % static_cast<std::size_t>(channels) != 0) {
        error = "Invalid WAV dimensions.";
        return false;
    }

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        error = "Cannot open WAV output: " + path.string();
        return false;
    }

    constexpr std::uint16_t bitsPerSample = 16;
    const auto dataBytes = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    const auto riffBytes = static_cast<std::uint32_t>(36 + dataBytes);
    const auto byteRate = static_cast<std::uint32_t>(sampleRate * channels * bitsPerSample / 8);
    const auto blockAlign = static_cast<std::uint16_t>(channels * bitsPerSample / 8);

    stream.write("RIFF", 4);
    writeValue(stream, riffBytes);
    stream.write("WAVEfmt ", 8);
    writeValue(stream, std::uint32_t{16});
    writeValue(stream, std::uint16_t{1});
    writeValue(stream, static_cast<std::uint16_t>(channels));
    writeValue(stream, static_cast<std::uint32_t>(sampleRate));
    writeValue(stream, byteRate);
    writeValue(stream, blockAlign);
    writeValue(stream, bitsPerSample);
    stream.write("data", 4);
    writeValue(stream, dataBytes);

    for (float sample : samples) {
        const auto clipped = std::clamp(sample, -1.0f, 1.0f);
        const auto pcm = static_cast<std::int16_t>(std::lrint(clipped * 32767.0f));
        writeValue(stream, pcm);
    }

    if (!stream) {
        error = "Failed while writing WAV data.";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace acestep

