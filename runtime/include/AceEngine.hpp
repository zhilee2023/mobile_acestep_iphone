#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace acestep {

struct GenerationRequest {
    std::string caption;
    std::string lyrics;
    int durationSeconds = 10;
    std::uint64_t seed = 42;
    int steps = 8;
};

class AceEngine {
public:
    bool initialize(const std::filesystem::path& modelDirectory, std::string& error);
    std::vector<float> generate(const GenerationRequest& request, std::string& error);
    bool ready() const noexcept;

private:
    std::filesystem::path modelDirectory_;
    bool ready_ = false;
};

}  // namespace acestep

