#pragma once

#include <filesystem>
#include <span>
#include <string>

namespace acestep {

bool writeFloatWav(
    const std::filesystem::path& path,
    std::span<const float> interleavedSamples,
    int sampleRate,
    int channels,
    std::string& error);

}  // namespace acestep

