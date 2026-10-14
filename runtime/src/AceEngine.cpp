#include "AceEngine.hpp"

#include <array>

namespace acestep {

bool AceEngine::initialize(const std::filesystem::path& modelDirectory, std::string& error) {
    static constexpr std::array<const char*, 3> requiredModels = {
        "text_encoder_w4.mnn", "dit_w4.mnn", "vae_decoder_w4.mnn"};

    for (const char* model : requiredModels) {
        const auto path = modelDirectory / model;
        if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) == 0) {
            error = "Missing or empty model: " + path.string();
            ready_ = false;
            return false;
        }
    }
    modelDirectory_ = modelDirectory;
    ready_ = true;
    error.clear();
    return true;
}

std::vector<float> AceEngine::generate(const GenerationRequest&, std::string& error) {
    if (!ready_) {
        error = "Engine is not initialized.";
        return {};
    }
    error = "MNN graph execution is intentionally gated until captured tensor contracts pass M1.";
    return {};
}

bool AceEngine::ready() const noexcept { return ready_; }

}  // namespace acestep

